#include <mpi.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <string>

// --- HELPER FUNCTIONS ---

inline int idx(int row, int col, int num_cols) {
    return row * num_cols + col;
}

std::vector<double> transpose(const std::vector<double>& matrix, int rows, int cols) {
    std::vector<double> result(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
    return result;
}

void log_stage(int rank, const std::string& msg) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\n[MASTER] >>> " << msg << "..." << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // 1. PARAMETERS
    int ROWS = 580; 
    int COLS = 130; 

    if (argc > 2) {
        ROWS = atoi(argv[1]);
        COLS = atoi(argv[2]);
    }

    if (world_rank == 0) {
        std::cout << "============================================" << std::endl;
        std::cout << " STARTING LAB WORK #6 (VARIANT 24)" << std::endl;
        std::cout << " Dimensions: " << ROWS << " rows x " << COLS << " cols" << std::endl;
        std::cout << " Process count: " << world_size << std::endl;
        std::cout << "============================================" << std::endl;
    }

    // 2. LOAD BALANCING
    std::vector<int> send_counts(world_size);
    std::vector<int> displs(world_size);

    int base_cols = COLS / world_size;
    int remainder = COLS % world_size;
    int current_disp = 0;

    for (int i = 0; i < world_size; i++) {
        send_counts[i] = base_cols + (i < remainder ? 1 : 0);
        displs[i] = current_disp;
        current_disp += send_counts[i];
    }

    int local_cols = send_counts[world_rank];

    // === STAGE 0: STATISTICS (MEMORY & OPS) ===
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        std::cout << "\n--- LOAD STATISTICS (for report) ---" << std::endl;
        std::cout << " Rank | Cols | Memory (KB) | Ops (FLOPs)" << std::endl;
        std::cout << "-----------------------------------------" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    long long elements_matrix = 4LL * local_cols * ROWS;
    long long elements_vector = 3LL * local_cols;
    long long total_elements = elements_matrix + elements_vector;
    double memory_kb = (total_elements * sizeof(double)) / 1024.0;

    long long flops_y1 = (long long)ROWS * local_cols * 2;
    long long flops_y2 = (long long)ROWS * local_cols * 2;
    long long flops_Y3 = (long long)ROWS * local_cols * 3;
    long long total_flops = flops_y1 + flops_y2 + flops_Y3;

    for (int i = 0; i < world_size; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == world_rank) {
            std::cout << " " << std::setw(4) << world_rank << " | "
                      << std::setw(4) << local_cols << " | "
                      << std::setw(11) << std::fixed << std::setprecision(2) << memory_kb << " | "
                      << std::setw(10) << total_flops << std::endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // === STAGE 2: MEMORY ALLOCATION ===
    log_stage(world_rank, "Stage 2: Memory allocation");

    std::vector<double> A, A1, B2, C2, full_A2; 
    std::vector<double> b, b1, c1;     

    std::vector<double> local_A_T(local_cols * ROWS);
    std::vector<double> local_A1_T(local_cols * ROWS);
    std::vector<double> local_B2_T(local_cols * ROWS);
    std::vector<double> local_C2_T(local_cols * ROWS);

    std::vector<double> local_b(local_cols);
    std::vector<double> local_b1(local_cols);
    std::vector<double> local_c1(local_cols);

    // === STAGE 3: DATA GENERATION ===
    log_stage(world_rank, "Stage 3: Data generation");

    if (world_rank == 0) {
        A.resize(ROWS * COLS); A1.resize(ROWS * COLS);
        B2.resize(ROWS * COLS); C2.resize(ROWS * COLS);
        full_A2.resize(ROWS * COLS);
        b.resize(COLS); b1.resize(COLS); c1.resize(COLS);

        for (int j = 0; j < COLS; ++j) {
            int i_math = j + 1;
            if (i_math % 2 == 0) b[j] = 24.0 / (i_math * i_math + 4.0);
            else b[j] = 24.0;
            
            b1[j] = 1.0; c1[j] = 1.0;
            for (int i = 0; i < ROWS; ++i) {
                A[idx(i, j, COLS)] = 1.0;
                A1[idx(i, j, COLS)] = 1.0;
                B2[idx(i, j, COLS)] = 1.0;
                full_A2[idx(i, j, COLS)] = 1.0; 
                C2[idx(i, j, COLS)] = 24.0 / ((i + 1) + 3.0 * (j + 1) * (j + 1));
            }
        }
        A = transpose(A, ROWS, COLS);
        A1 = transpose(A1, ROWS, COLS);
        B2 = transpose(B2, ROWS, COLS);
        C2 = transpose(C2, ROWS, COLS);
    } else {
        full_A2.resize(ROWS * COLS);
    }

    double start_time = MPI_Wtime();

    // === STAGE 4: SCATTER ===
    log_stage(world_rank, "Stage 4: Scatter & Bcast");
    
    std::vector<int> sc_counts(world_size), sc_displs(world_size);
    for(int i=0; i<world_size; ++i) {
        sc_counts[i] = send_counts[i] * ROWS;
        sc_displs[i] = displs[i] * ROWS;
    }

    MPI_Scatterv(A.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, local_A_T.data(), local_cols * ROWS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(A1.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, local_A1_T.data(), local_cols * ROWS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B2.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, local_B2_T.data(), local_cols * ROWS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(C2.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, local_C2_T.data(), local_cols * ROWS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(b.data(), send_counts.data(), displs.data(), MPI_DOUBLE, local_b.data(), local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b1.data(), send_counts.data(), displs.data(), MPI_DOUBLE, local_b1.data(), local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(c1.data(), send_counts.data(), displs.data(), MPI_DOUBLE, local_c1.data(), local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(full_A2.data(), ROWS * COLS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // === STAGE 5: COMPUTATION ===
    log_stage(world_rank, "Stage 5: Computation");

    // y1
    std::vector<double> local_y1(ROWS, 0.0);
    for (int k = 0; k < local_cols; ++k) {
        double val_b = local_b[k];
        for (int r = 0; r < ROWS; ++r) local_y1[r] += local_A_T[idx(k, r, ROWS)] * val_b;
    }
    std::vector<double> y1(ROWS);
    MPI_Allreduce(local_y1.data(), y1.data(), ROWS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // y2
    std::vector<double> local_y2(ROWS, 0.0);
    for (int k = 0; k < local_cols; ++k) {
        double val_vec = local_b1[k] - (24.0 * local_c1[k]);
        for (int r = 0; r < ROWS; ++r) local_y2[r] += local_A1_T[idx(k, r, ROWS)] * val_vec;
    }
    std::vector<double> y2(ROWS);
    MPI_Allreduce(local_y2.data(), y2.data(), ROWS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Y3
    std::vector<double> local_Y3_T(local_cols * ROWS);
    for (int k = 0; k < local_cols; ++k) {
        for (int r = 0; r < ROWS; ++r) {
            double val_B = local_B2_T[idx(k, r, ROWS)];
            double val_C = local_C2_T[idx(k, r, ROWS)];
            int global_col = displs[world_rank] + k;
            double val_A = full_A2[idx(r, global_col, COLS)];
            local_Y3_T[idx(k, r, ROWS)] = val_A * (val_B + 24.0 * val_C);
        }
    }

    // === STAGE 6: GATHER ===
    log_stage(world_rank, "Stage 6: Gather");
    std::vector<double> Y3_T;
    if (world_rank == 0) Y3_T.resize(ROWS * COLS);
    MPI_Gatherv(local_Y3_T.data(), local_cols * ROWS, MPI_DOUBLE, Y3_T.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // === STAGE 7: FINAL FORMULA ===
    log_stage(world_rank, "Stage 7: Final Formula");

    if (world_rank == 0) {
        std::vector<double> Y3 = transpose(Y3_T, COLS, ROWS);
        
        std::vector<double> RightPart(ROWS, 0.0);
        for(int i=0; i<ROWS; ++i) {
            double dot = 0.0;
            for(int j=0; j<COLS; ++j) dot += Y3[idx(i, j, COLS)] * y1[j];
            RightPart[i] = dot + y2[i];
        }

        std::vector<double> Y3_sq(ROWS * COLS);
        for(int i=0; i<ROWS*COLS; ++i) Y3_sq[i] = Y3[i] * Y3[i];

        std::vector<double> LeftPart(COLS, 0.0);
        for(int j=0; j<COLS; ++j) {
            double dot = 0.0;
            for(int i=0; i<ROWS; ++i) dot += y2[i] * Y3_sq[idx(i, j, COLS)];
            LeftPart[j] = dot + y1[j];
        }

        double final_x = 0.0;
        int min_dim = std::min(ROWS, COLS);
        for(int i=0; i<min_dim; ++i) final_x += LeftPart[i] * RightPart[i];

        double end_time = MPI_Wtime();

        std::cout << "\n============================================" << std::endl;
        std::cout << " RESULTS" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << " Time: " << (end_time - start_time) * 1000.0 << " ms" << std::endl;
        std::cout << " Final Result (X): " << std::scientific << final_x << std::endl;
        std::cout << "============================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}