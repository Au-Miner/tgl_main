#include <iostream>
#include <string>
#include <cstdlib>
#include <random>
#include <omp.h>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

typedef int NodeIDType;
typedef int EdgeIDType;
typedef float TimeStampType;

// 时序图块，用来存储每一层每一个快照的采样结果的
class TemporalGraphBlock
{
    public:
        // 不同的行id对应着不同的采样根节点，row.size()=根节点数
        std::vector<NodeIDType> row;
        // 不同的列id对应着不同的边，col.size()=新采样的节点数
        std::vector<NodeIDType> col;
        std::vector<EdgeIDType> eid;
        std::vector<TimeStampType> ts;
        std::vector<TimeStampType> dts;
        // 这个nodes感觉是存了从一开始的根节点到当前层经过的所有节点（和快照无关）
        std::vector<NodeIDType> nodes;
        // 该层该快照采样期间总共有多少节点（重复节点也会额外加上），该层该快照采样期间的根节点数（这里的根节点=从一开始的根节点到上一层经过的所有节点）
        NodeIDType dim_in, dim_out;
        // 存的是总共ptr更新的时间，用于测试
        double ptr_time = 0;
        // 存的是第0个线程的查找s_search和e_search时间，用于测试
        double search_time = 0;
        // 存的是第0个线程的采样时间，用于测试
        double sample_time = 0;
        // 存的是总的时间，用于测试
        double tot_time = 0;
        // 存的是combine的时间，用于测试
        double coo_time = 0;

        TemporalGraphBlock(){}

        TemporalGraphBlock(std::vector<NodeIDType> &_row, std::vector<NodeIDType> &_col,
                           std::vector<EdgeIDType> &_eid, std::vector<TimeStampType> &_ts,
                           std::vector<TimeStampType> &_dts, std::vector<NodeIDType> &_nodes,
                           NodeIDType _dim_in, NodeIDType _dim_out) :
                           row(_row), col(_col), eid(_eid), ts(_ts), dts(_dts),
                           nodes(_nodes), dim_in(_dim_in), dim_out(_dim_out) {}
};

class ParallelSampler
{
    public:
        std::vector<EdgeIDType> indptr;
        std::vector<EdgeIDType> indices;
        std::vector<EdgeIDType> eid;    // edge的id，它的顺序和时间先后顺序保持一致
        std::vector<TimeStampType> ts;
        NodeIDType num_nodes;
        EdgeIDType num_edges;
        int num_thread_per_worker;      // 每个worker的线程数
        int num_workers;                // worker数
        int num_threads;                // 总共线程数
        int num_layers;                 // 总共采样多少层
        std::vector<int> num_neighbors; // 一个整数列表，指示每层中采样的邻居数
        bool recent;                    // 是否对最近的邻居进行采样
        bool prop_time;                 // False或True，指定在为其多跳邻居采样时使用根节点的时间戳的位置
        int num_history;                // 要采样的快照数
        TimeStampType window_duration;  // 每个快照的时间长度，0表示无限长（用于非基于快照的方法）
        // std::vector<EdgeIDType>::size_type 等价于 size_type，不理解为什么这样子设计
        // 在节点对应的时间限制下，处在第i个快照中第j个节点的所有边的出现时间大于等于节点时间限制的第一条边的指针
        // ts_ptr可以o(1)找到s_search和e_search
        std::vector<std::vector<std::vector<EdgeIDType>::size_type>> ts_ptr;
        omp_lock_t *ts_ptr_lock;
        // ret大小=采样层数*采样快照数
        std::vector<TemporalGraphBlock> ret;

        ParallelSampler(std::vector<EdgeIDType> &_indptr, std::vector<EdgeIDType> &_indices,
                        std::vector<EdgeIDType> &_eid, std::vector<TimeStampType> &_ts,
                        int _num_thread_per_worker, int _num_workers, int _num_layers,
                        std::vector<int> &_num_neighbors, bool _recent, bool _prop_time,
                        int _num_history, TimeStampType _window_duration) :
                        indptr(_indptr), indices(_indices), eid(_eid), ts(_ts), prop_time(_prop_time),
                        num_thread_per_worker(_num_thread_per_worker), num_workers(_num_workers),
                        num_layers(_num_layers), num_neighbors(_num_neighbors), recent(_recent),
                        num_history(_num_history), window_duration(_window_duration)
        {
            omp_set_num_threads(num_thread_per_worker * num_workers);
            num_threads = num_thread_per_worker * num_workers;
            num_nodes = indptr.size() - 1;
            num_edges = indices.size();
            // 当并行更新指针时，多个线程可能共享具有不同时间戳的相同目标节点，这会导致竞争条件。我们为每个节点添加细粒度锁，以避免指针在这种情况下被多次提前
            ts_ptr_lock = (omp_lock_t *)malloc(num_nodes * sizeof(omp_lock_t));
            for (int i = 0; i < num_nodes; i++)
                omp_init_lock(&ts_ptr_lock[i]);
            // 对ts_ptr进行初始化，设置每一快照内每个节点第一条边的指针都为以当前节点为起点最早的边的指针（应该是存所处indices的下标吧）
            // ts_ptr最外层大小设置为n+1
            ts_ptr.resize(num_history + 1);
            for (auto it = ts_ptr.begin(); it != ts_ptr.end(); it++)
            {
                // ts_ptr中间层大小设置为v
                it->resize(indptr.size() - 1);
#pragma omp parallel for
                for (auto itt = indptr.begin(); itt < indptr.end() - 1; itt++)
                    (*it)[itt - indptr.begin()] = *itt;
            }
        }

        // 重置tr_ptr的状态
        void reset()
        {
            for (auto it = ts_ptr.begin(); it != ts_ptr.end(); it++)
            {
                it->resize(indptr.size() - 1);
#pragma omp parallel for
                for (auto itt = indptr.begin(); itt < indptr.end() - 1; itt++)
                    (*it)[itt - indptr.begin()] = *itt;
            }
        }

        // 因为所有采样节点有对应时间戳限制，所以将ts_ptr中第slc个快照每个采样节点第一条边的指针进行修改（向后移动）
        void update_ts_ptr(int slc, std::vector<NodeIDType> &root_nodes,
                           std::vector<TimeStampType> &root_ts, float offset)
        {
#pragma omp parallel for schedule(static, int(ceil(static_cast<float>(root_nodes.size()) / num_threads)))
            // 枚举所有要更新的节点
            for (std::vector<NodeIDType>::size_type i = 0; i < root_nodes.size(); i++)
            {
                NodeIDType n = root_nodes[i];
                omp_set_lock(&(ts_ptr_lock[n]));
                // 递增当前节点n在slc层下的初始边在indices中的下标
                for (std::vector<EdgeIDType>::size_type j = ts_ptr[slc][n]; j < indptr[n + 1]; j++)
                {
                    // std::cout << "comparing " << ts[j] << " with " << root_ts[i] << std::endl;
                    // 如果当前边出现的时刻大于等于了要求的时刻
                    if (ts[j] > (root_ts[i] + offset - 1e-7f))
                    {
                        // 如果下标不是一开始的位置，那么更新该节点当前层的下标
                        if (j != ts_ptr[slc][n])
                            ts_ptr[slc][n] = j - 1;
                        break;
                    }
                    // 如果下标已经到达了当前节点的最后一个下标，那么直接将该层下标移动到这
                    if (j == indptr[n + 1] - 1)
                    {
                        ts_ptr[slc][n] = j;
                    }
                }
                omp_unset_lock(&(ts_ptr_lock[n]));
            }
        }

        // 采样当前边，将其存入临时变量中
        inline void add_neighbor(std::vector<NodeIDType> *_row, std::vector<NodeIDType> *_col,
                                 std::vector<EdgeIDType> *_eid, std::vector<TimeStampType> *_ts,
                                 std::vector<TimeStampType> *_dts, std::vector<NodeIDType> *_nodes,
                                 EdgeIDType &k, TimeStampType &src_ts, int &row_id)
        {
            _row->push_back(row_id);
            _col->push_back(_nodes->size());
            _eid->push_back(eid[k]);
            if (prop_time)
                _ts->push_back(src_ts);
            else
                _ts->push_back(ts[k]);
            _dts->push_back(src_ts - ts[k]);
            _nodes->push_back(indices[k]);
            // _row.push_back(0);
            // _col.push_back(0);
            // _eid.push_back(0);
            // if (prop_time)
            //     _ts.push_back(src_ts);
            // else
            //     _ts.push_back(10000);
            // _nodes.push_back(100);
        }

        // 将临时变量中的数据合并到对应层数，对应快照的ret中
        inline void combine_coo(TemporalGraphBlock &_ret, std::vector<NodeIDType> **_row,
                                std::vector<NodeIDType> **_col,
                                std::vector<EdgeIDType> **_eid,
                                std::vector<TimeStampType> **_ts,
                                std::vector<TimeStampType> **_dts,
                                std::vector<NodeIDType> **_nodes,
                                std::vector<int> &_out_nodes)
        {
            std::vector<EdgeIDType> cum_row, cum_col;
            cum_row.push_back(0);
            cum_col.push_back(0);
            for (int tid = 0; tid < num_threads; tid++)
            {
                // std::cout<<tid<<" here "<<_out_nodes[tid]<<std::endl;
                cum_row.push_back(cum_row.back() + _out_nodes[tid]);
                cum_col.push_back(cum_col.back() + _col[tid]->size());
            }
            int num_root_nodes = _ret.nodes.size();
            _ret.row.resize(cum_col.back());
            _ret.col.resize(cum_col.back());
            _ret.eid.resize(cum_col.back());
            _ret.ts.resize(cum_col.back() + num_root_nodes);
            _ret.dts.resize(cum_col.back() + num_root_nodes);
            _ret.nodes.resize(cum_col.back() + num_root_nodes);
#pragma omp parallel for schedule(static, 1)
            for (int tid = 0; tid < num_threads; tid++)
            {
                // transform(first,last,result,op)：对[first,last)执行op结果存放在[result,result+last-first)
                // 将每一个线程内采样的节点的行号加上前面总的行数
                std::transform(_row[tid]->begin(), _row[tid]->end(), _row[tid]->begin(),
                               [&](auto &v){ return v + cum_row[tid]; });
                // 将每一个线程内采样的节点的列号加上前面总的列数和根节点数
                std::transform(_col[tid]->begin(), _col[tid]->end(), _col[tid]->begin(),
                               [&](auto &v){ return v + cum_col[tid] + num_root_nodes; });
                // copy(first,end,result)：将[first,last)内的元素复制到[result,result+last-first)中
                // 将每一个线程内采样的节点的6个属性分别添加到_ret中对应的数组中（_ts,_dts和_nodes因为之前存放过根节点，所以这里要额外+num_root_nodes）
                std::copy(_row[tid]->begin(), _row[tid]->end(), _ret.row.begin() + cum_col[tid]);
                std::copy(_col[tid]->begin(), _col[tid]->end(), _ret.col.begin() + cum_col[tid]);
                std::copy(_eid[tid]->begin(), _eid[tid]->end(), _ret.eid.begin() + cum_col[tid]);
                std::copy(_ts[tid]->begin(), _ts[tid]->end(), _ret.ts.begin() + cum_col[tid] + num_root_nodes);
                std::copy(_dts[tid]->begin(), _dts[tid]->end(), _ret.dts.begin() + cum_col[tid] + num_root_nodes);
                std::copy(_nodes[tid]->begin(), _nodes[tid]->end(), _ret.nodes.begin() + cum_col[tid] + num_root_nodes);
                delete _row[tid];
                delete _col[tid];
                delete _eid[tid];
                delete _ts[tid];
                delete _dts[tid];
                delete _nodes[tid];
            }
            _ret.dim_in = _ret.nodes.size();
            _ret.dim_out = cum_row.back();
        }

        void sample_layer(std::vector<NodeIDType> &_root_nodes, std::vector<TimeStampType> &_root_ts,
                          int neighs, bool use_ptr, bool from_root)
                          // from_root：是否为第一层采样
        {
            double t_s = omp_get_wtime();
            std::vector<NodeIDType> *root_nodes;
            std::vector<TimeStampType> *root_ts;
            if (from_root)
            {
                root_nodes = &_root_nodes;
                root_ts = &_root_ts;
            }
            double t_ptr_s = omp_get_wtime();
            // 第num_history个图应该对应最新图
            if (use_ptr)
                update_ts_ptr(num_history, *root_nodes, *root_ts, 0);
            ret[0].ptr_time += omp_get_wtime() - t_ptr_s;
            for (int i = 0; i < num_history; i++)
            {
                if (!from_root)
                {
                    // 如果非第一层采样，那么获取上一层对应的快照的采样节点（ret在对每层采样之前都会增大快照数大小，所以多-num_history就是获取之前的所有节点了）
                    root_nodes = &(ret[ret.size() - 1 - i - num_history].nodes);
                    root_ts = &(ret[ret.size() - 1 - i - num_history].ts);
                }
                TimeStampType offset = -i * window_duration;
                t_ptr_s = omp_get_wtime();
                // 如果使用ptr且使用快照策略，那么对第num_history - 1 - i个快照更新每个采样节点出现第一条边的指针
                if ((use_ptr) && (std::abs(window_duration) > 1e-7f))
                    update_ts_ptr(num_history - 1 - i, *root_nodes, *root_ts, offset - window_duration);
                ret[0].ptr_time += omp_get_wtime() - t_ptr_s;
                std::vector<NodeIDType> *_row[num_threads];
                std::vector<NodeIDType> *_col[num_threads];
                std::vector<EdgeIDType> *_eid[num_threads];
                std::vector<TimeStampType> *_ts[num_threads];
                std::vector<TimeStampType> *_dts[num_threads];
                std::vector<NodeIDType> *_nodes[num_threads];
                // _out_node：第i个线程采样的根节点数
                std::vector<int> _out_node(num_threads, 0);
                int reserve_capacity = int(ceil((*root_nodes).size() / num_threads)) * neighs;
#pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    unsigned int loc_seed = tid;
                    _row[tid] = new std::vector<NodeIDType>;
                    _col[tid] = new std::vector<NodeIDType>;
                    _eid[tid] = new std::vector<EdgeIDType>;
                    _ts[tid] = new std::vector<TimeStampType>;
                    _dts[tid] = new std::vector<TimeStampType>;
                    _nodes[tid] = new std::vector<NodeIDType>;
                    _row[tid]->reserve(reserve_capacity);
                    _col[tid]->reserve(reserve_capacity);
                    _eid[tid]->reserve(reserve_capacity);
                    _ts[tid]->reserve(reserve_capacity);
                    _dts[tid]->reserve(reserve_capacity);
                    _nodes[tid]->reserve(reserve_capacity);
// #pragma omp critical
//                     std::cout<<tid<<" sampling: "<<root_nodes->size()<<" "<<int(ceil((*root_nodes).size() / num_threads))<<std::endl;
// #pragma omp for schedule：把for循环一块一块的分给线程
#pragma omp for schedule(static, int(ceil(static_cast<float>((*root_nodes).size()) / num_threads)))
                    for (std::vector<NodeIDType>::size_type j = 0; j < (*root_nodes).size(); j++)
                    {
                        NodeIDType n = (*root_nodes)[j];
                        // if (tid == 16)
                        //     std::cout << _out_node[tid] << " " <<j << " " << n << std::endl;
                        TimeStampType nts = (*root_ts)[j];
                        EdgeIDType s_search, e_search;
                        if (use_ptr)
                        {
                            // 枚举indices从s_search到e_search表示：n号节点在第num_history - i个快照中大于等于时间限制出现的第一条边之前一个快照周期中的所有边
                            // 查找范围应该是(s_search,e_search]
                            s_search = ts_ptr[num_history - 1 - i][n];
                            e_search = ts_ptr[num_history - i][n];
                        }
                        else
                        {
                            // search for start and end pointer
                            double t_search_s = omp_get_wtime();
                            if (num_history == 1)
                            {
                                // TGAT style
                                // 二分查找n在整个时态图中在时间限制下能走到的最后一条边
                                // 查找范围应该是[s_search,e_search]
                                s_search = indptr[n];
                                auto e_it = std::upper_bound(ts.begin() + indptr[n],
                                                             ts.begin() + indptr[n + 1], nts);
                                e_search = std::max(int(e_it - ts.begin()) - 1, s_search);
                            }
                            else
                            {
                                // DySAT style
                                // s_it表示上一个快照中>时间限制的第一条边前面的指针；nts + offset - window_duration=当前节点时间限制-快照id*快照周期
                                // 查找范围应该是(s_search,e_search]
                                auto s_it = std::upper_bound(ts.begin() + indptr[n],
                                                             ts.begin() + indptr[n + 1],
                                                             nts + offset - window_duration);
                                s_search = std::max(int(s_it - ts.begin()) - 1, indptr[n]);
                                // e_it表示当前快照中>时间限制的第一条边前面的指针（如果i==0代表最新图所更新的那部分）
                                auto e_it = std::upper_bound(ts.begin() + indptr[n],
                                                             ts.begin() + indptr[n + 1], nts + offset);
                                e_search = std::max(int(e_it - ts.begin()) - 1, s_search);
                            }
                            if (tid == 0)
                                ret[0].search_time += omp_get_wtime() - t_search_s;
                        }
                        // std::cout << n << " " << s_search << " " << e_search << std::endl;
                        double t_sample_s = omp_get_wtime();
                        if ((recent) || (e_search - s_search < neighs))
                        {
                            // no sampling, pick recent neighbors
                            // 从e_search开始直接拿边
                            for (EdgeIDType k = e_search; k > std::max(s_search, e_search - neighs); k--)
                            {
                                // 如果当前边的时间小于当前的时间限制（nts + offset：当前快照的截止时间），那么将采样该点
                                // 疑问：我觉得前面的[]和()有待商榷，但最后还是用下面的if来特判了
                                if (ts[k] < nts + offset - 1e-7f)
                                {
                                    add_neighbor(_row[tid], _col[tid], _eid[tid], _ts[tid],
                                                 _dts[tid], _nodes[tid], k, nts, _out_node[tid]);
                                }
                            }
                        }
                        else
                        {
                            // random sampling within ptr
                            for (int _i = 0; _i < neighs; _i++)
                            {
                                EdgeIDType picked = s_search + rand_r(&loc_seed) % (e_search - s_search + 1);
                                if (ts[picked] < nts + offset - 1e-7f)
                                {
                                    add_neighbor(_row[tid], _col[tid], _eid[tid], _ts[tid],
                                                 _dts[tid], _nodes[tid], picked, nts, _out_node[tid]);
                                }
                            }
                        }
                        _out_node[tid] += 1;
                        if (tid == 0)
                            ret[0].sample_time += omp_get_wtime() - t_sample_s;
                    }
                }
                double t_coo_s = omp_get_wtime();
                // 将根节点的id和时间戳存入ret中（从后往前存）
                ret[ret.size() - 1 - i].ts.insert(ret[ret.size() - 1 - i].ts.end(),
                                                  root_ts->begin(), root_ts->end());
                ret[ret.size() - 1 - i].nodes.insert(ret[ret.size() - 1 - i].nodes.end(),
                                                     root_nodes->begin(), root_nodes->end());
                ret[ret.size() - 1 - i].dts.resize(root_nodes->size());
                combine_coo(ret[ret.size() - 1 - i], _row, _col, _eid, _ts, _dts, _nodes, _out_node);
                ret[0].coo_time += omp_get_wtime() - t_coo_s;
            }
            ret[0].tot_time += omp_get_wtime() - t_s;
        }

        void sample(std::vector<NodeIDType> &root_nodes, std::vector<TimeStampType> &root_ts)
        {
            // a weird bug, dgl library seems to modify the total number of threads
            omp_set_num_threads(num_threads);
            ret.resize(0);
            bool first_layer = true;
            bool use_ptr = false;
            for (int i = 0; i < num_layers; i++)
            {
                // ret的大小不断变大，每次变大快照数
                ret.resize(ret.size() + num_history);
                // 如果是第一层采样，或者多层采样选取根节点时间戳且只采样一个快照，或者最近采样原则，那么使用ptr
                // 不为第一层采样，选取根节点时间戳时必须要快照数为1才能使用ptr：不明白为什么要限制快照数必须为1
                // 原先考虑的是：root_ts会改变，在ptr_update的时候判断条件是>(root_ts[i] + offset - 1e-7f)，因此与“选取根节点时间戳”矛盾了，但其实在add_neighbor的时候还是用的根节点时间戳啊
                if ((first_layer) || ((prop_time) && num_history == 1) || (recent))
                {
                    first_layer = false;
                    use_ptr = true;
                }
                else
                    use_ptr = false;
                if (i==0)
                    sample_layer(root_nodes, root_ts, num_neighbors[i], use_ptr, true);
                else
                    sample_layer(root_nodes, root_ts, num_neighbors[i], use_ptr, false);
            }
        }
};

template<typename T>
inline py::array vec2npy(const std::vector<T> &vec)
{
    // need to let python garbage collector handle C++ vector memory 
    // see https://github.com/pybind/pybind11/issues/1042
    auto v = new std::vector<T>(vec);
    auto capsule = py::capsule(v, [](void *v)
                               { delete reinterpret_cast<std::vector<T> *>(v); });
    return py::array(v->size(), v->data(), capsule);
    // return py::array(vec.size(), vec.data());
}

PYBIND11_MODULE(sampler_core, m)
{
    py::class_<TemporalGraphBlock>(m, "TemporalGraphBlock")
        .def(py::init<std::vector<NodeIDType> &, std::vector<NodeIDType> &,
                      std::vector<EdgeIDType> &, std::vector<TimeStampType> &,
                      std::vector<TimeStampType> &, std::vector<NodeIDType> &,
                      NodeIDType, NodeIDType>())
        .def("row", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.row); })
        .def("col", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.col); })
        .def("eid", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.eid); })
        .def("ts", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.ts); })
        .def("dts", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.dts); })
        .def("nodes", [](const TemporalGraphBlock &tgb) { return vec2npy(tgb.nodes); })
        .def("dim_in", [](const TemporalGraphBlock &tgb) { return tgb.dim_in; })
        .def("dim_out", [](const TemporalGraphBlock &tgb) { return tgb.dim_out; })
        .def("tot_time", [](const TemporalGraphBlock &tgb) { return tgb.tot_time; })
        .def("ptr_time", [](const TemporalGraphBlock &tgb) { return tgb.ptr_time; })
        .def("search_time", [](const TemporalGraphBlock &tgb) { return tgb.search_time; })
        .def("sample_time", [](const TemporalGraphBlock &tgb) { return tgb.sample_time; })
        .def("coo_time", [](const TemporalGraphBlock &tgb) { return tgb.coo_time; });
    py::class_<ParallelSampler>(m, "ParallelSampler")
        .def(py::init<std::vector<EdgeIDType> &, std::vector<EdgeIDType> &,
                      std::vector<EdgeIDType> &, std::vector<TimeStampType> &,
                      int, int, int, std::vector<int> &, bool, bool,
                      int, TimeStampType>())
        .def("sample", &ParallelSampler::sample)
        .def("reset", &ParallelSampler::reset)
        .def("get_ret", [](const ParallelSampler &ps) { return ps.ret; });
}