#if 0

set -eux

 #ASAN_SYMBOLIZER_PATH=$ROCM_PATH/llvm/bin/llvm-symbolizer \
 #$ROCM_PATH/llvm/bin/clang++ -fsanitize=address -fno-omit-frame-pointer -std=c++11 -O0 -g -o rccl-log-loader rccl-log-loader.cpp

g++ -fopenmp -std=c++11 -O3 -o rccl-log-post-process rccl-log-post-process.cpp

input=rccl-log-64

taskset -c 0-63 ./rccl-log-post-process $input.data $input.json

rm -rf $input.json.xz
taskset -c 0-63 xz -T8 --keep $input.json

exit 0
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <cassert>
#include <sstream>
#include <limits>
#include <algorithm>

#include "rccl-log-loader.h"

struct rank_obj {
    unsigned rank;
    std::vector<comm_obj> comms;
    std::vector<op_obj> ops;

    // Map of comms address to comms hash
    std::map<size_t,size_t> comms_address_to_hash;

    rank_obj(unsigned rank, unsigned ncomms, unsigned nops) : rank(rank), comms(ncomms), ops(nops) {}
};

const char * txt_header = R"E0E1(
{
  "schemaVersion": 1,
  "record_shapes": 1,
  "with_stack": 1,
  "traceEvents": [
)E0E1";

const char * txt_entry = R"E0E1(
  {
    "ph": "X", "cat": "rccl_op", "name": "%s", "pid": %u, "tid": %u,
    "ts": %lf, "dur": %lf,
    "args": {
      "opcount": %u, "type": "%s", "count": %lu, "kB": %.3f, "GB/s": %.3f, "ranks": [%s]
    }
  },
)E0E1";

const char * txt_process_label = R"E0E1(
  {
    "name": "process_name", "ph": "M", "ts": 0.0, "pid": %u, "tid": 0,
    "args": {
      "name": "Communicator %u"
    }
  },
)E0E1";

const char * txt_thread_label = R"E0E1(
  {
    "name": "thread_name", "ph": "M", "ts": 0.0, "pid": %u, "tid": %u,
    "args": {
      "name": "rank %04u"
    }
  },
)E0E1";

const char * txt_footer = R"E0E1(
  ],
  "traceName": "rccl-post-process",
  "displayTimeUnit": "ms",
}
)E0E1";


int main (int argc, char *argv[]) {

    if (argc < 3) {
        printf("File not specified. Execute as: ./rccl-log-post-process input-file.data output-file.json\n");
        exit(-1);
    }

    FILE *fp = nullptr;
    if(!(fp = fopen(argv[1],"rb"))) {
        printf("Can't read result file %s.\n", argv[1]);
        exit(-1);
    }

    // nranks
    // rank
    // ncomms
    // nops
    // comm0...commn
    // op0...opn

    unsigned nranks;
    while(fread(&nranks, sizeof(unsigned), 1, fp) != 1);

    // Map from comms hash to global rank
    typedef std::vector<unsigned> global_rank_ty;

    std::map<size_t,global_rank_ty> comm_hash_to_global_rank;
    std::map<size_t,std::string> comm_hash_to_global_rank_str;
    std::map<size_t,unsigned> comm_hash_to_global_rank_sort;

    std::vector<rank_obj> rank_data;
    for(unsigned r = 0; r<nranks; ++r) {

        unsigned rank, ncomms, nops;
        while(fread(&rank, sizeof(unsigned), 1, fp) != 1);
        while(fread(&ncomms, sizeof(unsigned), 1, fp) != 1);
        while(fread(&nops, sizeof(unsigned), 1, fp) != 1);

        rank_data.emplace_back(rank, ncomms, nops);
        auto &rd = rank_data.back();

        for(auto &c : rd.comms) {
            while(fread(&c, sizeof(comm_obj), 1, fp) != 1);

            rd.comms_address_to_hash[c.comm] = c.commId;

            // Initialize comm hash
            if (!comm_hash_to_global_rank.count(c.commId))
                comm_hash_to_global_rank.emplace(std::make_pair(c.commId, global_rank_ty(c.nranks,-1u)));
        
            comm_hash_to_global_rank[c.commId][c.rank] = rank;
        }
        for(auto &o : rd.ops)
            while(fread(&o, sizeof(op_obj), 1, fp) != 1);
    }

    fclose(fp);

    // Select an order for the communicators:
    // - per number of ranks 
    // - per the first rank of the communicator
    size_t first_comm = 0;
    {
        unsigned ncomms = comm_hash_to_global_rank.size();
        std::vector<size_t> comm_hash;
        for(auto &m : comm_hash_to_global_rank)
            comm_hash.push_back(m.first);

        std::sort(comm_hash.begin(),comm_hash.end(),[&comm_hash_to_global_rank](size_t a, size_t b){

            if (a == b)
                return false;

            auto &ma = comm_hash_to_global_rank[a];
            auto &mb = comm_hash_to_global_rank[b];

            if (ma.size() > mb.size())
                return true;

            for(unsigned i=0; i<ma.size(); ++i){
                if (ma[i] < mb[i])
                    return true;
            }

            return false;
        });

        first_comm = comm_hash.front();

        unsigned i = 0;
        for(auto c: comm_hash)
            comm_hash_to_global_rank_sort[c] = ++i;
    }

    // If we have a communicator for all ranks, let's use the first op to calculate deviation between timmings.
    // We use a linear interpolation betwen the first and last op.

    std::vector<std::pair<double,double>> deviation(nranks, {0.0, 0.0});
    if (first_comm && comm_hash_to_global_rank[first_comm].size() ==  nranks) {

        double t1i=0.0, t1e=0.0;
        unsigned opi=0, ope=0;

        {
            auto &rd = rank_data.front();
            for(auto &o: rd.ops) {

                if (rd.comms_address_to_hash[o.comm] != first_comm)
                    continue;

                if (o.opcount < 1)
                    continue;

                if (strstr(o.collname, "AllReduce")  == nullptr)
                    continue;
                    
                if (opi == 0) {
                    opi = o.opcount;
                    t1i = o.te;
                }

                ope = o.opcount;
                t1e = o.te;
            }
        }

        if(opi && ope && opi != ope) {
            for (auto &rd : rank_data) {
                if (rd.rank == 0)
                    continue;

                double t2i=0.0, t2e=0.0;

                for (auto &o: rd.ops) {
                    if (o.opcount == opi) t2i = o.te;
                    if (o.opcount == ope) t2e = o.te;
                }

                printf("%lf-%lf %lf-%lf\n", t2e, t2i, t1e, t1i);

                double a = t2i-t1i;
                double b = t2e-t1e;
                double x1 = (b-a)/(t2e-t2i);
                double x2 = a - t2i*(b-a)/(t2e-t2i);

                printf("%lf %lf\n", x1*t2i+x2, t2i - (x1*t2i+x2));
                deviation[rd.rank] = {x1, x2};
                break;
            }
        }
    }

    // Generate the strings with lists of ranks per communicator.
    for (const auto& m : comm_hash_to_global_rank) {
        std::stringstream ss;
        for(unsigned r : m.second)
            ss << r << ",";

        comm_hash_to_global_rank_str[m.first] = ss.str();
        //printf("Comm [%u] info 0x%lx --> [%s]\n", comm_hash_to_global_rank_sort[m.first], m.first, ss.str().c_str());
    }

    fp = nullptr;
    if(!(fp = fopen(argv[2],"w"))) {
        printf("Can't open result file %s.\n", argv[1]);
        exit(-1);
    }

    fprintf(fp,txt_header);

    for(auto &rd: rank_data) {
        for(auto &o: rd.ops) {

            bool invalid_ts = o.ts <= 0.0 || o.ts >= std::numeric_limits<double>::max();
            bool invalid_te = o.te <= 0.0 || o.te <= std::numeric_limits<double>::min();

            if(invalid_ts || invalid_te)
                continue;

            size_t comm_hash = rd.comms_address_to_hash[o.comm];

            auto &dev_par = deviation[rd.rank];
            double dev_ts = dev_par.first * o.ts + dev_par.second;
            double dev_te = dev_par.first * o.te + dev_par.second;

            double ts = (o.ts - dev_ts)*1e6;
            double te = (o.te - dev_te)*1e6; 
            double dur = te-ts;
            double bytes = (double)o.count*rccl_type_to_bytes(o.datatype);

            fprintf(fp,
                txt_entry, 
                /*name*/ o.collname,
                /*pid*/ comm_hash_to_global_rank_sort[comm_hash],
                /*tid*/ rd.rank,
                /*ts*/ ts,
                /*dur*/ dur,
                /*opcount*/ o.opcount,
                /*type*/ rccl_type_to_str(o.datatype),
                /*count*/ o.count,
                /*kB*/ bytes*1e-3,
                /*GB/s*/ bytes/(dur*1e3),
                /*ranks*/ comm_hash_to_global_rank_str[comm_hash].c_str());
        }

        // fprintf(fp,
        //         txt_thread_label,
        //         /*pid*/ 1,
        //         /*tid*/ rd.rank+1,
        //         /*rank*/ rd.rank);
    }

    for (const auto& m : comm_hash_to_global_rank_sort)
        fprintf(fp,
                txt_process_label,
                /*pid*/ m.second,
                /*comm idx*/ m.second);

    fprintf(fp,txt_footer);

    fclose(fp);
    return 0;
}