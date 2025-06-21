#pragma once
#define STR_MAX_HOSTNAME 16
#define STR_MAX_COLLNAME 16
#define STR_MAX_KERNELNAME 64

enum op_type {
    OP_TYPE_COLL = 0,
    OP_TYPE_SEND = 1,
    OP_TYPE_RECV = 2,
};

struct comm_obj {

    char hostname[STR_MAX_HOSTNAME] = {0};
    unsigned pid, tid, devid;
    size_t comm;
    unsigned rank, nranks, cudaDev, nvmlDev, busId;
    size_t commId, localSize, bytes;
    unsigned core;

    comm_obj() {}

    static comm_obj* generator(char *line) {
        // Communicator info
        // e.g. x1000c3s5b0n0:906353:907286 [2] NCCL INFO ncclCommInitRank comm 0xfe08fa0 rank 170 nranks 256 cudaDev 2 nvmlDev 2 busId c9000 commId 0xc2f42310f0fc7fea localSize 296 used 368870624 bytes on core 19 - Init COMPLETE
        
        if (strstr(line, " Init COMPLETE") == nullptr)
            return nullptr;

        auto *tt = new comm_obj();
        auto &t = *tt;

        int conversions = sscanf(line,
            "%15[^:]:%u:%u [%u] NCCL INFO ncclCommInitRank comm %lx rank %u nranks %u cudaDev %u nvmlDev %u busId %x commId %lx localSize %ld used %ld bytes on core %d - Init COMPLETE",
            &t.hostname[0],
            &t.pid, &t.tid, &t.devid,
            &t.comm,
            &t.rank, &t.nranks, &t.cudaDev, &t.nvmlDev, &t.busId,
            &t.commId, &t.localSize, &t.bytes,
            &t.core);

        if (conversions == 14) {
            //printf("sfantao -->%s<--\n", line);
            //tt->pretty_print();
            return tt;
        }

        printf("Failed to match Comm: %s\n", line);

        delete tt;
        return nullptr;
    }

        
    void pretty_print() {
        printf(
            "proc = %s:%u:%u\n"
            "devid = %u\n"
            "comm = 0x%lx\n"
            "rank = %u\n"
            "nranks = %u\n"
            "cudaDev = %u\n"
            "nvmlDev = %u\n"
            "busId = %x\n"
            "commId = 0x%lx\n"
            "localSize = %lu\n"
            "bytes = %lu\n"
            "core = %u\n",
            hostname,
            pid, tid, devid,
            comm,
            rank, nranks, cudaDev, nvmlDev, busId,
            commId, localSize, bytes,
            core);
    }
};

struct op_obj {

    char hostname[STR_MAX_HOSTNAME] = {0};
    unsigned pid, tid, devid;
    char collname[STR_MAX_COLLNAME] = {0};
    unsigned opcount;
    size_t sendbuff, recvbuff;
    unsigned count, datatype, op, root;
    size_t comm;
    unsigned nranks;
    size_t stream;
    unsigned task, globalrank;

    double ts = -1.0;
    double te = -1.0;

    op_obj() {}

    static op_obj* generator(char *line) {
        // Communicator info
        // e.g. x1000c3s5b0n0:906353:906353 [2] NCCL INFO AllReduce: opCount 0 sendbuff 0x7f9196a00200 recvbuff 0x7f9196a00200 count 1 datatype 7 op 0 root 0 comm 0xfe08fa0 [nranks=256] stream 0xfb70f80 task 0 globalrank 170
        
        if (strstr(line, " opCount ") == nullptr)
            return nullptr;
        
        auto *tt = new op_obj();
        auto &t = *tt;

        int conversions = 0;
        conversions = sscanf(line,
            "%15[^:]:%u:%u [%u] NCCL INFO %15[^:]: opCount %x sendbuff %lx recvbuff %lx count %u datatype %u op %u root %u comm %lx [nranks=%u] stream %lx task %u globalrank %u",
            &t.hostname[0],
            &t.pid, &t.tid, &t.devid,
            &t.collname[0],
            &t.opcount,
            &t.sendbuff, &t.recvbuff,
            &t.count, &t.datatype, &t.op, &t.root,
            &t.comm,
            &t.nranks,
            &t.stream,
            &t.task, &t.globalrank);

        if (conversions == 17) {
            // printf("sfantao -->%s\n", line);
            // tt->pretty_print();
            return tt;
        }

        conversions = sscanf(line,
            "%15[^:]:%u:%u [%u] NCCL INFO %15[^:]: opCount %x sendbuff (nil) recvbuff %lx count %u datatype %u op %u root %u comm %lx [nranks=%u] stream %lx task %u globalrank %u",
            &t.hostname[0],
            &t.pid, &t.tid, &t.devid,
            &t.collname[0],
            &t.opcount,
            &t.recvbuff,
            &t.count, &t.datatype, &t.op, &t.root,
            &t.comm,
            &t.nranks,
            &t.stream,
            &t.task, &t.globalrank);

        t.sendbuff = 0;

        if (conversions == 16) {
            // printf("sfantao -->%s\n", line);
            // tt->pretty_print();
            return tt;
        }

        printf("Failed to match opCount: %s\n", line);

        delete tt;
        return nullptr;
    }

        
    void pretty_print() {
        printf(
            "proc = %s:%u:%u\n"
            "devid = %u\n"
            "collname = %s\n"
            "opcount = %x\n"
            "send/recv = 0x%lx/0x%lx\n"
            "count = %u\n"
            "datatype = %u\n"
            "op = %u\n"
            "root = %u\n"
            "comm = 0x%lx\n"
            "nranks = %u\n"
            "stream = 0x%lx\n"
            "task = 0x%u\n"
            "globalrank = 0x%u\n"
            "duration = %lf-%lf\n\n",
            hostname,
            pid, tid, devid,
            collname,
            opcount,
            sendbuff, recvbuff,
            count, datatype, op, root,
            comm,
            nranks,
            stream,
            task,
            globalrank,
            ts, te);

        return;
    }
};

struct kl_obj {

    char hostname[STR_MAX_HOSTNAME] = {0};
    unsigned pid, tid, devid;
    double timestamp;
    unsigned opcount, hwid;
    char kernelname[STR_MAX_KERNELNAME] = {0};
    unsigned busid, nranks;
    size_t comm;
    op_type type = op_type::OP_TYPE_COLL;

    kl_obj() {}

    static kl_obj* generator(char *line) {
        // Communicator info
        // e.g. x1000c3s5b0n0:906353:908380 [2] NCCL INFO ## [951297.535389] [170:00-00:00] 000000 KL HWID   302510 ncclDevFunc_AllReduce_RING_LL_Sum_f32 nw 4 bi 0 nc 1 root 0 busId c9000 nRanks 256 comm 0x12345
        //      x1000c0s2b0n0:160290:164392 [0] NCCL INFO ## [122210.650274] [00:00-01:c0] 000000-000000 KL HWID   302c00 ncclDevFunc_SendRecv_RING_SIMPLE_Sum_i8 Send 1 -> 1/0 ConnIdx/LL 1/0 -> Recv 1 nc 4 cb 1 busId c1000 nRanks 2 comm 0x2b901420

        if (strstr(line, " KL ") == nullptr)
            return nullptr;

        auto *tt = new kl_obj();
        auto &t = *tt;

        unsigned dummy, nw, bi, nc, root, cb;

        int conversions = 0;

        conversions = sscanf(line,
            "%15[^:]:%u:%u [%u] NCCL INFO ## [%lf] [%x:%x-%x:%x] %x KL HWID   %x %63[^ ] nw %u bi %u nc %u root %u busId %x nRanks %u comm %lx",
            &t.hostname[0],
            &t.pid, &t.tid, &t.devid,
            &t.timestamp,
            &dummy,&dummy,&dummy,&dummy,
            &t.opcount, &t.hwid,
            &t.kernelname[0],
            &nw, &bi, &nc, &root, &t.busid, &t.nranks,
            &t.comm);

        if (conversions == 19) {
            // printf("sfantao -->%s<--\n", line);
            // tt->pretty_print();
            return tt;
        }

        unsigned op1, op2;
        conversions = sscanf(line,
            "%15[^:]:%u:%u [%u] NCCL INFO ## [%lf] [%x:%x-%x:%x] %x-%x KL HWID   %x %63[^ ] Send %u -> %u/%u ConnIdx/LL %u/%u -> Recv %u nc %u cb %u busId %x nRanks %u comm %lx",
            &t.hostname[0],
            &t.pid, &t.tid, &t.devid,
            &t.timestamp,
            &dummy,&dummy,&dummy,&dummy,
            &op1, &op2, &t.hwid,
            &t.kernelname[0],
            &dummy,&dummy,&dummy,&dummy,&dummy,&dummy,
            &nc, &cb, &t.busid, &t.nranks,
            &t.comm);

        if (conversions == 24) {
            t.opcount = (op2>0) ? op2 : op1;
            t.type = (op2>0) ? op_type::OP_TYPE_RECV : op_type::OP_TYPE_SEND;

            // printf("sfantao -->%s<--\n", line);
            // tt->pretty_print();
            return tt;
        }

        printf("Failed to match KL: %s\n", line);

        delete tt;
        return nullptr;
    }

        
    void pretty_print() {
        printf(
            "proc = %s:%u:%u\n"
            "devid = %u\n"
            "timestamp = %lf\n"
            "opcount = %x\n"
            "hwid = 0x%x\n"
            "kernelname = %s\n"
            "busid = %x\n"
            "nranks = %u\n"
            "comm = 0x%lx\n"
            "type = %u\n",
            hostname,
            pid, tid, devid,
            timestamp,
            opcount, hwid,
            kernelname,
            busid, nranks,
            comm,
            type);
        return;
    }
};

struct ke_obj {
    char hostname[STR_MAX_HOSTNAME] = {0};
    unsigned pid, tid, devid;
    double timestamp;
    unsigned opcount, busid, nranks;
    size_t comm;
    op_type type = op_type::OP_TYPE_COLL;

    ke_obj() {}

    static ke_obj* generator(char *line) {
        // Communicator info
        // e.g. x1000c3s5b0n0:906353:908380 [2] NCCL INFO ## [951297.641095] [170:00-00:40] 000000 KE busId c9000 nRanks 256 comm 0x2b901420
        //      x1000c0s2b0n0:1602905:1604392 [0] NCCL INFO ## [1222100.655884] [00:00-01:80] 000000-000000 KE busId c1000 nRanks 2 comm 0x2b901420
        
        if (strstr(line, " KE ") == nullptr)
            return nullptr;
        
        auto *tt = new ke_obj();
        auto &t = *tt;

        unsigned dummy;

        int conversions = 0;
        conversions = sscanf(line,
            "%15[^:]:%u:%u [%u] NCCL INFO ## [%lf] [%x:%x-%x:%x] %x KE busId %x nRanks %u comm %lx",
            &t.hostname[0],
            &t.pid, &t.tid, &t.devid,
            &t.timestamp,
            &dummy,&dummy,&dummy,&dummy,
            &t.opcount, &t.busid, &t.nranks,
            &t.comm);

        if (conversions == 13) {
            // printf("sfantao -->%s<--\n", line);
            // tt->pretty_print();
            return tt;
        }

        unsigned op2, op1;
        conversions = sscanf(line,
            "%15[^:]:%u:%u [%u] NCCL INFO ## [%lf] [%x:%x-%x:%x] %x-%x KE busId %x nRanks %u comm %lx",
            &t.hostname[0],
            &t.pid, &t.tid, &t.devid,
            &t.timestamp,
            &dummy,&dummy,&dummy,&dummy,
            &op1, &op2, &t.busid, &t.nranks,
            &t.comm);

        if (conversions == 14) {
            t.opcount = (op2>0) ? op2 : op1;
            t.type = (op2>0) ? op_type::OP_TYPE_RECV : op_type::OP_TYPE_SEND;

            // printf("sfantao -->%s<--\n", line);
            // tt->pretty_print();
            return tt;
        }

        printf("Failed to match KE: %s\n", line);

        delete tt;
        return nullptr;
    }
        
    void pretty_print() {
        printf(
            "proc = %s:%u:%u\n"
            "devid = %u\n"
            "timestamp = %lf\n"
            "opcount = %x\n"
            "busid = %x\n"
            "nranks = %u\n"
            "comm = 0x%lx\n"
            "type = %u\n",
            hostname,
            pid, tid, devid,
            timestamp,
            opcount,
            busid, nranks,
            comm,
            type);
        return;
    }
};
