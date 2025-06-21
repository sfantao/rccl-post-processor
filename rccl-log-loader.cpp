#if 0

set -eux

 #ASAN_SYMBOLIZER_PATH=$ROCM_PATH/llvm/bin/llvm-symbolizer \
 #$ROCM_PATH/llvm/bin/clang++ -fsanitize=address -fno-omit-frame-pointer -std=c++11 -O0 -g -o rccl-log-loader rccl-log-loader.cpp

g++ -fopenmp -std=c++11 -O3 -o rccl-log-loader rccl-log-loader.cpp

OMP_NUM_THREADS=8 taskset -c 0-63 ./rccl-log-loader ./sfantao-nccllog2-results-16/nccl-logs/sfantao-rccl-rank*.txt
OMP_NUM_THREADS=8 taskset -c 0-63 ./rccl-log-loader ./sfantao-nccllog2-results-32/nccl-logs/sfantao-rccl-rank*.txt

exit 0
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <limits>
#include <cassert>

#include "rccl-log-loader.h"

struct rank_obj {
    unsigned rank;
    const char *fname;
    std::vector<comm_obj*> comms;
    std::vector<op_obj*> ops;
};

bool load_rank_data (rank_obj &rd) {
    FILE *fp = nullptr;

    fp = fopen(rd.fname, "r");
    if (!fp) {
        printf("Error opening file %s.\n", rd.fname);
        return true;
    }

    char * line = nullptr;
    size_t linelen = 0;
    size_t len;
    size_t readlen;
    size_t total = 0;

    auto &comms = rd.comms;
    auto &ops = rd.ops;
    std::vector<kl_obj*> kls;
    std::vector<ke_obj*> kes;

    while ((readlen = getline(&line, &len, fp)) != -1) {
        if (auto *t = comm_obj::generator(line)) {
            comms.push_back(t);
            continue;
        }
        if (auto *t = op_obj::generator(line)) {
            ops.push_back(t);
            continue;
        }
        if (auto *t = kl_obj::generator(line)) {
            kls.push_back(t);
            continue;
        }
        if (auto *t = ke_obj::generator(line)) {
            kes.push_back(t);
            continue;
        }
    }

    if (line)
        free(line);

    fclose(fp);

    printf("Creating map...\n");

    // Create map 
    // comm -> opcount -> send/recv/coll -> (ts,te)
    typedef std::pair<double,double> ty_times; 
    typedef std::map<op_type,ty_times> ty_types;
    typedef std::map<unsigned,ty_types> ty_opcounts;
    typedef std::map<size_t,ty_opcounts> ty_comms;
    ty_comms mapper;

    for (auto &k : kls) {

        bool needs_init = 
            !mapper.count(k->comm) || 
            !mapper[k->comm].count(k->opcount) ||
            !mapper[k->comm][k->opcount].count(k->type);

        if(needs_init)
            mapper[k->comm][k->opcount][k->type] = ty_times(std::numeric_limits<double>::max(),
                                                            std::numeric_limits<double>::min());

        auto &v = mapper[k->comm][k->opcount][k->type].first;
        v = std::min(v, k->timestamp);
    }

    for (auto &k : kes) {
        bool needs_init = 
            !mapper.count(k->comm) || 
            !mapper[k->comm].count(k->opcount) ||
            !mapper[k->comm][k->opcount].count(k->type);

        if(needs_init)
            mapper[k->comm][k->opcount][k->type] = ty_times(std::numeric_limits<double>::max(),
                                                            std::numeric_limits<double>::min());

        auto &v = mapper[k->comm][k->opcount][k->type].second;
        v = std::max(v, k->timestamp);
    }

    printf("Setting ops timmings...\n");
    size_t incomplete_ops = 0;
    for (auto &o : ops) {

        if(o->opcount == 0)
            continue;

        op_type type = op_type::OP_TYPE_COLL;

        if (strstr(o->collname, "Send") != nullptr)
            type = op_type::OP_TYPE_SEND;
        else if (strstr(o->collname, "Recv") != nullptr)
            type = op_type::OP_TYPE_RECV;

        auto &v = mapper[o->comm][o->opcount][type];
        o->ts = v.first;
        o->te = v.second;

        bool invalid_ts = o->ts <= 0.0 || o->ts >= std::numeric_limits<double>::max();
        bool invalid_te = o->te <= 0.0 || o->ts <= std::numeric_limits<double>::min();

        if (invalid_ts || invalid_te) {
            ++incomplete_ops;

            if (!invalid_ts)
                o->te = o->ts;
            if (!invalid_te)
                o->ts = o->te;

            //printf("Invalid timmings for:\n");
            //o->pretty_print();
        }

    }

    printf("[%u]--> Missing information %0.3f%%.\n", rd.rank, (double)incomplete_ops*100.0/ops.size());
    printf("[%u]--> comms %ld\n",rd.rank,comms.size());
    printf("[%u]--> ops %ld\n",rd.rank,ops.size());
    printf("[%u]--> kls %ld\n",rd.rank,kls.size());
    printf("[%u]--> kes %ld\n",rd.rank,kes.size());

    for (auto *v: kls) delete v;
    for (auto *v: kes) delete v;

    return false;
}

int main (int argc, char *argv[]) {

    if (argc < 2) {
        printf("No files specified.\n");
        return -1;
    }

    int nranks = argc-1;
    std::vector<rank_obj> rank_data(nranks);

    #pragma omp parallel for
    for(unsigned r=0; r<nranks; ++r) {
        const char *fname = argv[r+1];
        unsigned rank;

        const char *pos = strstr(fname, "-rank");

        if (pos == nullptr) {
            printf("Unable to identify rank info in file %s\n",fname);
            continue;
        }

        int conversions = sscanf(pos,"-rank%u", &rank);

        if (conversions != 1) {
            printf("Unable to determine rank for file %s\n",fname);
            continue;
        }

        if (rank > nranks) {
            printf("Invalid rank %u out of %u for file %s\n",rank, nranks, fname);
            continue;
        }

        auto &rd = rank_data[rank];
        rd.rank = rank;
        rd.fname = fname;

        printf("Loading data for rank %u of %u from file %s\n",rank,nranks,fname);

        if (load_rank_data(rd)) {
            printf("Error loading data for rank %u out of %u from file %s\n",rank, nranks, fname);
            continue;
        }
    }

    FILE *fp = nullptr;

    char fname[64];
    sprintf(fname,"rccl-log-%u.data",nranks);

    if(!(fp = fopen(fname,"wb"))) {
        printf("Can't create result file\n");
        exit(-1);
    }

    // nranks
    // rank
    // ncomms
    // nops
    // comm0...commn
    // op0...opn

    while(fwrite(&nranks, sizeof(int), 1, fp) != 1);
    for( auto &rd : rank_data) {
        unsigned rank = rd.rank;
        unsigned ncomms = rd.comms.size();
        unsigned nops = rd.ops.size();

        while(fwrite(&rank, sizeof(unsigned), 1, fp) != 1);
        while(fwrite(&ncomms, sizeof(unsigned), 1, fp) != 1);
        while(fwrite(&nops, sizeof(unsigned), 1, fp) != 1);

        for(auto *c : rd.comms)
            while(fwrite(c, sizeof(comm_obj), 1, fp) != 1);
        for(auto *o : rd.ops)
            while(fwrite(o, sizeof(op_obj), 1, fp) != 1);
    }

    fclose(fp);

    return 0;
}