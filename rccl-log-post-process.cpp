#if 0

set -eux

 #ASAN_SYMBOLIZER_PATH=$ROCM_PATH/llvm/bin/llvm-symbolizer \
 #$ROCM_PATH/llvm/bin/clang++ -fsanitize=address -fno-omit-frame-pointer -std=c++11 -O0 -g -o rccl-log-loader rccl-log-loader.cpp

g++ -fopenmp -std=c++11 -O3 -o rccl-log-post-process rccl-log-post-process.cpp

taskset -c 0-63 ./rccl-log-post-process rccl-log-64.data rccl-log-64.json
taskset -c 0-63 xz -T8 --keep rccl-log-64.json

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
      "opcount": %u, "type": %u, "count": %lu, "ranks": [%s]
    }
  },
)E0E1";

const char * txt_process_label = R"E0E1(
  {
    "name": "process_name", "ph": "M", "ts": 0.0, "pid": %u, "tid": 0,
    "args": {
      "name": "%s"
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
        printf("File not specified\n");
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

    for (const auto& m : comm_hash_to_global_rank) {
        std::stringstream ss;
        for(unsigned r : m.second)
            ss << r << ",";

        comm_hash_to_global_rank_str[m.first] = ss.str();
        printf("Comm info 0x%lx --> [%s]\n", m.first, ss.str().c_str());
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
            bool invalid_te = o.te <= 0.0 || o.ts <= std::numeric_limits<double>::min();

            if(invalid_ts || invalid_te)
                continue;

            fprintf(fp,
                txt_entry, 
                /*name*/ o.collname,
                /*pid*/ 1,
                /*tid*/ rd.rank+1,
                /*ts*/ o.ts*1e6,
                /*dur*/ (o.te-o.ts)*1e6,
                /*opcount*/ o.opcount,
                /*type*/ o.datatype,
                /*count*/ o.count,
                /*ranks*/ comm_hash_to_global_rank_str[rd.comms_address_to_hash[o.comm]].c_str());
        }

        fprintf(fp,
                txt_thread_label,
                /*pid*/ 1,
                /*tid*/ rd.rank+1,
                /*rank*/ rd.rank);
    }

    fprintf(fp,
            txt_process_label,
            /*pid*/ 1,
            /*name*/ "main RCCL process");
    fprintf(fp,txt_footer);

    fclose(fp);
    return 0;
}