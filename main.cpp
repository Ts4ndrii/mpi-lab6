/*
 * Лабораторна робота №6. Варіант 23.
 * Матриця: 580 рядків (ROWS) x 130 стовпців (COLS).
 * Додано: Автоматичний розрахунок складності та пам'яті.
 */

#include <mpi.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <string>

// --- ДОПОМІЖНІ ФУНКЦІЇ ---

inline int idx(int row, int col, int num_cols) {
    return row * num_cols + col;
}

// Транспонування
std::vector<double> transpose(const std::vector<double>& matrix, int rows, int cols) {
    std::vector<double> result(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
    return result;
}

// Логування етапів
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

    // 1. ПАРАМЕТРИ
    int ROWS = 580; 
    int COLS = 130; 

    if (argc > 2) {
        ROWS = atoi(argv[1]);
        COLS = atoi(argv[2]);
    }

    if (world_rank == 0) {
        std::cout << "============================================" << std::endl;
        std::cout << " ЗАПУСК ЛАБОРАТОРНОЇ РОБОТИ №6 (ВАРІАНТ 23)" << std::endl;
        std::cout << " Розмірність: " << ROWS << " рядків x " << COLS << " стовпців" << std::endl;
        std::cout << " Кількість процесів: " << world_size << std::endl;
        std::cout << "============================================" << std::endl;
    }

    // 2. РОЗПОДІЛ НАВАНТАЖЕННЯ
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

    // === ЕТАП 0: РОЗРАХУНОК ТА ВИВІД СТАТИСТИКИ (НОВЕ) ===
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        std::cout << "\n--- СТАТИСТИКА НАВАНТАЖЕННЯ (для звіту) ---" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Розрахунок пам'яті (Тільки розподілені дані, без врахування дубльованої A2)
    // 4 матриці (A, A1, B2, C2) * local_cols * ROWS
    // 3 вектори (b, b1, c1) * local_cols
    long long elements_matrix = 4LL * local_cols * ROWS;
    long long elements_vector = 3LL * local_cols;
    long long total_elements = elements_matrix + elements_vector;
    double memory_kb = (total_elements * sizeof(double)) / 1024.0;

    // Розрахунок операцій (FLOPs estimate)
    // y1 = A*b:  ROWS * local_cols * 2 (mult + add)
    // y2 = ...:  ROWS * local_cols * 2 + local_cols*2 (vector prep)
    // Y3 = ...:  ROWS * local_cols * 3 (add, mult, mult)
    long long flops_y1 = (long long)ROWS * local_cols * 2;
    long long flops_y2 = (long long)ROWS * local_cols * 2;
    long long flops_Y3 = (long long)ROWS * local_cols * 3;
    long long total_flops = flops_y1 + flops_y2 + flops_Y3;

    // Виводимо статистику для кожного процесора (синхронізовано)
    for (int i = 0; i < world_size; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == world_rank) {
            std::cout << "[Rank " << std::setw(2) << world_rank << "] "
                      << "Стовпців: " << std::setw(3) << local_cols << " | "
                      << "Пам'ять: " << std::fixed << std::setprecision(2) << memory_kb << " KB | "
                      << "Операцій: " << total_flops << " FLOPs" << std::endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // === ЕТАП 2: ВИДІЛЕННЯ ПАМ'ЯТІ ===
    log_stage(world_rank, "Етап 2: Виділення пам'яті");

    std::vector<double> A, A1, B2, C2, full_A2; 
    std::vector<double> b, b1, c1;     

    std::vector<double> local_A_T(local_cols * ROWS);
    std::vector<double> local_A1_T(local_cols * ROWS);
    std::vector<double> local_B2_T(local_cols * ROWS);
    std::vector<double> local_C2_T(local_cols * ROWS);

    std::vector<double> local_b(local_cols);
    std::vector<double> local_b1(local_cols);
    std::vector<double> local_c1(local_cols);

    // === ЕТАП 3: ГЕНЕРАЦІЯ ===
    log_stage(world_rank, "Етап 3: Генерація даних");

    if (world_rank == 0) {
        A.resize(ROWS * COLS); A1.resize(ROWS * COLS);
        B2.resize(ROWS * COLS); C2.resize(ROWS * COLS);
        full_A2.resize(ROWS * COLS);
        b.resize(COLS); b1.resize(COLS); c1.resize(COLS);

        for (int j = 0; j < COLS; ++j) {
            b[j] = ((j + 1) % 2 == 0) ? (24.0 / ((j + 1) * (j + 1) + 4)) : 24.0;
            b1[j] = 1.0; c1[j] = 1.0;
            for (int i = 0; i < ROWS; ++i) {
                A[idx(i, j, COLS)] = 1.0;
                A1[idx(i, j, COLS)] = 1.0;
                B2[idx(i, j, COLS)] = 1.0;
                full_A2[idx(i, j, COLS)] = 1.0; 
                C2[idx(i, j, COLS)] = 24.0 / (i + 1 + 3.0 * (j + 1) * (j + 1));
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

    // === ЕТАП 4: РОЗСИЛКА ===
    log_stage(world_rank, "Етап 4: Scatter & Bcast");
    
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

    // === ЕТАП 5: ОБЧИСЛЕННЯ ===
    log_stage(world_rank, "Етап 5: Обчислення");

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

    // === ЕТАП 6: ЗБІР ===
    log_stage(world_rank, "Етап 6: Gather");
    std::vector<double> Y3_T;
    if (world_rank == 0) Y3_T.resize(ROWS * COLS);
    MPI_Gatherv(local_Y3_T.data(), local_cols * ROWS, MPI_DOUBLE, Y3_T.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // === ЕТАП 7: ФІНАЛ ===
    log_stage(world_rank, "Етап 7: Фінальна формула");

    if (world_rank == 0) {
        std::vector<double> Y3 = transpose(Y3_T, COLS, ROWS);
        
        double D = 0.0; // y1 * y2
        for(int i=0; i<ROWS; ++i) D += y1[i] * y2[i];

        double term1_sum = 0.0, term2_sum = 0.0, term3_sum = 0.0;
        for(auto val : Y3) {
            term1_sum += val;
            term2_sum += val * val;
            term3_sum += val * val * val;
        }

        // X = Y3 + (y1*y2)*Y3^2 + ...
        double FinalResult = term1_sum + (D * term2_sum) + (D * term3_sum) + (D * term1_sum * term1_sum);

        double end_time = MPI_Wtime();

        std::cout << "\n============================================" << std::endl;
        std::cout << " РЕЗУЛЬТАТИ" << std::endl;
        std::cout << " Час: " << (end_time - start_time) * 1000.0 << " ms" << std::endl;
        std::cout << " y1*y2: " << D << std::endl;
        std::cout << " Фінальний результат (X): " << std::scientific << FinalResult << std::endl;
        std::cout << "============================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}