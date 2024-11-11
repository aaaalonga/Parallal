#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>

void fill_rand(float* M, int n) {
    for (int i = 0; i < n; i++) {
        M[i] = (float)rand() / RAND_MAX;
    }
}

int op_encode(MPI_Op op) {
    if (op == MPI_SUM) {
        return 1;
    }
    if (op == MPI_MAX) {
        return 2;
    }
    return 0;
}

const char* op_name(MPI_Op op) {
    if (op == MPI_OP_NULL) {
        return "null";
    }
    if (op == MPI_SUM) {
        return "sum";
    }
    if (op == MPI_MAX) {
        return "max";
    }
    return "unknwon";
}

MPI_Op op_decode(int op) {
    if (op == 1) {
        return MPI_SUM;
    }
    if (op == 2) {
        return MPI_MAX;
    }
    return MPI_OP_NULL;
}

static int idx_fmt(int idx, int np) {
    idx %= np;
    return idx < 0 ? idx + np : idx;
}
static int idx_at(int idx, int count, int np) {
    return count / np * idx;
}
static int idx_sz(int idx, int count, int np) {
    int lo = idx_at(idx, count, np);
    int up = idx_at(idx + 1, count, np);
    up = up < count ? up : count;
    return up - lo;
}
static void range_info(int* at, int* sz, int idx, int count, int np) {
    idx = idx_fmt(idx, np);
    *at = idx_at(idx, count, np);
    *sz = idx_sz(idx, count, np);
}
static void reduce_float(float* dst, const float* src, int n, MPI_Op op) {
    if (op == MPI_SUM) {
        for (int i = 0; i < n; i++) {
            dst[i] += src[i];
        }
    }
    if (op == MPI_MAX) {
        for (int i = 0; i < n; i++) {
            dst[i] = dst[i] > src[i] ? dst[i] : src[i];
        }
    }
}

int RING_Allreduce(
    const void* sendbuf, void* recvbuf, int count,
    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    if (datatype != MPI_FLOAT) {
        printf("RING_Allreduce datatype error!\n");
        return -1;
    }
    if (op != MPI_MAX && op != MPI_SUM) {
        printf("RING_Allreduce op error!\n");
        return -1;
    }
    const float* sbuf = (const float*)sendbuf;
    float* rbuf = (float*)recvbuf;

    int pid, np;
    MPI_Comm_rank(comm, &pid);
    MPI_Comm_size(comm, &np);

    int prev = idx_fmt(pid - 1, np);
    int next = idx_fmt(pid + 1, np);
    MPI_Request reqs[2];
    MPI_Status stus[2];

    for (int i = 0; i < (np - 1) * 2; i++) {
        int send_at, send_sz, recv_at, recv_sz;
        range_info(&send_at, &send_sz, pid - i, count, np);
        range_info(&recv_at, &recv_sz, pid - i - 1, count, np);
        const float* sx = i == 0 ? &sbuf[send_at] : &rbuf[send_at];
        float* rx = &rbuf[recv_at];
        MPI_Isend(sx, send_sz, datatype, next, i, comm, &reqs[0]);
        MPI_Irecv(rx, recv_sz, datatype, prev, i, comm, &reqs[1]);
        // MPI_Waitall(2, reqs, stus);
        MPI_Wait(&reqs[1], &stus[1]);
        if (i < np - 1) {
            reduce_float(rx, &sbuf[recv_at], recv_sz, op);
        }
        MPI_Wait(&reqs[0], &stus[0]);
    }
    return 0;
}

int main(int argc, char** argv) {
    srand(123);
    int pid, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    int n = 0;
    int optag = 0;
    MPI_Op op = MPI_OP_NULL;
    if (pid == 0) {
        char* lab_n = getenv("AUTOSET");
        if (lab_n != NULL) {
            printf("using AUTOSET envp\n");
            n = 16 * 1024 * 1024;
            // n = 16;
            op = MPI_MAX;
        } else {
            while (n <= 0 || n % np != 0) {
                printf("input n:\n");

                if (scanf("%d", &n) != 1) {
                    printf("format error!\n");
                    exit(-1);
                }
                if (n % np != 0) {
                    printf("n %% np(%d) != 0!\n", np);
                    exit(-1);
                }
            }
            while (op == MPI_OP_NULL) {
                printf("input op:(sum or max)\n");
                char buf[10];
                if (scanf("%10s", buf) != 1) {
                    printf("format error!\n");
                    exit(-1);
                }
                buf[9] = '\0';
                if (strcmp(buf, "sum") == 0) {
                    op = MPI_SUM;
                } else if (strcmp(buf, "max") == 0) {
                    op = MPI_MAX;
                } else {
                    printf("format error!\n");
                    exit(-1);
                }
            }
        }
        if (n % np != 0) {
            n = (n + np - 1) / np * np;
        }
        if (op == MPI_OP_NULL) {
            printf("op error!\n");
            exit(-1);
        }
        printf("n: %d op: %s\n", n, op_name(op));
        optag = op_encode(op);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&optag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    op = op_decode(optag);

    float* A = NULL, * B = NULL, * C = NULL; // C = A * B
    double t0, t1, s0, s1;

    A = (float*)malloc(n * sizeof(float));
    B = (float*)malloc(n * sizeof(float));
    C = (float*)malloc(n * sizeof(float));

    fill_rand(A, n);
    MPI_Barrier(MPI_COMM_WORLD);

    t0 = MPI_Wtime();
    MPI_Allreduce(A, B, n, MPI_FLOAT, op, MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    s0 = t1 - t0;

    t0 = MPI_Wtime();
    RING_Allreduce(A, C, n, MPI_FLOAT, op, MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    s1 = t1 - t0;

    if (pid == 0) {
        printf(" mpi: %.1fms\n", s0 * 1000);
        printf("ring: %.1fms\n", s1 * 1000);
    }
    int fail = 0;
    for (int i = 0; i < n; i++) {
        float base = fmaxf(fabs(B[i]), 1.f);
        if (fabsf((B[i] - C[i]) / base) > 1e-4) {
            fail += 1;
        }
    }
    if (fail) {
        printf("check fail: %d pid %d\n", fail, pid);
    } else if (pid == 0) {
        printf("check success\n");
    }

    free(A);
    free(B);
    free(C);

    MPI_Finalize();
    return 0;
}
