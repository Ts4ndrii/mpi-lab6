/*
 * Лабораторна робота №6. Варіант 23.
 * Матриця: 580 рядків (ROWS) x 130 стовпців (COLS).
 * Розбиття: Вертикальне (Column-wise).
 * Кількість процесорів: 13 (по 10 стовпців на процес).
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

// Транспонування (для зручної розсилки стовпців)
std::vector<double> transpose(const std::vector<double>& matrix, int rows, int cols) {
    std::vector<double> result(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
    return result;
}

// Функція для синхронізованого виводу логів
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

    // Параметри згідно завдання
    int ROWS = 580; 
    int COLS = 130; 

    // Можливість перевизначити через аргументи
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

    // === ЕТАП 1: РОЗПОДІЛ НАВАНТАЖЕННЯ ===
    log_stage(world_rank, "Етап 1: Розрахунок розподілу навантаження");
    
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
    
    // Кожен процес звітує про своє навантаження
    // (Щоб не було каші, робимо це по черзі)
    for (int i = 0; i < world_size; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == world_rank) {
            std::cout << "  -> Rank " << world_rank << " отримав " << local_cols << " стовпців." << std::endl;
        }
    }

    // === ЕТАП 2: ВИДІЛЕННЯ ПАМ'ЯТІ ===
    log_stage(world_rank, "Етап 2: Виділення пам'яті");

    // Глобальні змінні (Rank 0)
    std::vector<double> A, A1, B2, C2, full_A2; 
    std::vector<double> b, b1, c1;     

    // Локальні буфери
    std::vector<double> local_A_T(local_cols * ROWS);
    std::vector<double> local_A1_T(local_cols * ROWS);
    std::vector<double> local_B2_T(local_cols * ROWS);
    std::vector<double> local_C2_T(local_cols * ROWS);

    std::vector<double> local_b(local_cols);
    std::vector<double> local_b1(local_cols);
    std::vector<double> local_c1(local_cols);

    // === ЕТАП 3: ГЕНЕРАЦІЯ ДАНИХ ===
    log_stage(world_rank, "Етап 3: Генерація та транспонування даних (на Master)");

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
        std::cout << "  [Master] Дані згенеровано. Виконую транспонування для розсилки..." << std::endl;
        
        A = transpose(A, ROWS, COLS);
        A1 = transpose(A1, ROWS, COLS);
        B2 = transpose(B2, ROWS, COLS);
        C2 = transpose(C2, ROWS, COLS);
    } else {
        full_A2.resize(ROWS * COLS); // Інші процеси повинні мати місце під A2
    }

    double start_time = MPI_Wtime();

    // === ЕТАП 4: РОЗСИЛКА (SCATTER) ===
    log_stage(world_rank, "Етап 4: Розсилка даних (Scatter & Bcast)");
    
    std::vector<int> sc_counts(world_size), sc_displs(world_size);
    for(int i=0; i<world_size; ++i) {
        sc_counts[i] = send_counts[i] * ROWS;
        sc_displs[i] = displs[i] * ROWS;
    }

    MPI_Scatterv(A.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, 
                 local_A_T.data(), local_cols * ROWS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(A1.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, 
                 local_A1_T.data(), local_cols * ROWS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B2.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, 
                 local_B2_T.data(), local_cols * ROWS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(C2.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, 
                 local_C2_T.data(), local_cols * ROWS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(b.data(), send_counts.data(), displs.data(), MPI_DOUBLE, 
                 local_b.data(), local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b1.data(), send_counts.data(), displs.data(), MPI_DOUBLE, 
                 local_b1.data(), local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(c1.data(), send_counts.data(), displs.data(), MPI_DOUBLE, 
                 local_c1.data(), local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(full_A2.data(), ROWS * COLS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // === ЕТАП 5: ОБЧИСЛЕННЯ ===
    log_stage(world_rank, "Етап 5: Паралельні обчислення");

    // --- y1 ---
    std::vector<double> local_y1(ROWS, 0.0);
    for (int k = 0; k < local_cols; ++k) {
        double val_b = local_b[k];
        for (int r = 0; r < ROWS; ++r) {
            local_y1[r] += local_A_T[idx(k, r, ROWS)] * val_b;
        }
    }
    std::vector<double> y1(ROWS);
    MPI_Allreduce(local_y1.data(), y1.data(), ROWS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(world_rank == 0) std::cout << "  -> y1 обчислено." << std::endl;

    // --- y2 ---
    std::vector<double> local_y2(ROWS, 0.0);
    for (int k = 0; k < local_cols; ++k) {
        double val_vec = local_b1[k] - (24.0 * local_c1[k]);
        for (int r = 0; r < ROWS; ++r) {
            local_y2[r] += local_A1_T[idx(k, r, ROWS)] * val_vec;
        }
    }
    std::vector<double> y2(ROWS);
    MPI_Allreduce(local_y2.data(), y2.data(), ROWS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(world_rank == 0) std::cout << "  -> y2 обчислено." << std::endl;

    // --- Y3 ---
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
    if(world_rank == 0) std::cout << "  -> Частини Y3 обчислено." << std::endl;

    // === ЕТАП 6: ЗБІР РЕЗУЛЬТАТІВ ===
    log_stage(world_rank, "Етап 6: Збір результатів (Gather)");
    
    std::vector<double> Y3_T;
    if (world_rank == 0) Y3_T.resize(ROWS * COLS);

    MPI_Gatherv(local_Y3_T.data(), local_cols * ROWS, MPI_DOUBLE,
                Y3_T.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // === ЕТАП 7: ФІНАЛЬНИЙ ВИРАЗ ===
    log_stage(world_rank, "Етап 7: Обчислення фінального виразу (Rank 0)");

    if (world_rank == 0) {
        // Відновлюємо форму Y3
        std::vector<double> Y3 = transpose(Y3_T, COLS, ROWS);
        
        // Скалярний добуток y1 * y2
        double D = 0.0;
        for(int i=0; i<ROWS; ++i) D += y1[i] * y2[i];
        std::cout << "  -> Скалярний добуток (y1*y2) = " << D << std::endl;

        // Обчислення складного виразу з Лаб 4:
        // X = Y3 + (y1*y2)*Y3^2 + (y1*y2)*Y3^3 ...
        // Оскільки матриця прямокутна, "піднесення до степеня" інтерпретуємо як поелементне,
        // а фінальний результат зводимо до скаляра (суми елементів), як у прикладі Лаб 4.
        
        double term1_sum = 0.0; // Сума елементів Y3
        double term2_sum = 0.0; // Сума елементів Y3^2
        double term3_sum = 0.0; // Сума елементів Y3^3

        for(auto val : Y3) {
            term1_sum += val;
            term2_sum += val * val;
            term3_sum += val * val * val;
        }

        // Фінальна формула (апроксимація логіки Лаб 4 для скалярного результату):
        // X_final = Sum(Y3) + D * Sum(Y3^2) + D * Sum(Y3^3) + D * Sum(Y3)*Sum(Y3)
        // (Останній доданок імітує y2*Y3 * y1*Y3)
        
        double FinalResult = term1_sum + (D * term2_sum) + (D * term3_sum) + (D * term1_sum * term1_sum);

        double end_time = MPI_Wtime();

        std::cout << "\n============================================" << std::endl;
        std::cout << " РЕЗУЛЬТАТИ ВИКОНАННЯ" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << " Час виконання: " << std::fixed << std::setprecision(4) 
                  << (end_time - start_time) * 1000.0 << " ms" << std::endl;
        std::cout << "--------------------------------------------" << std::endl;
        std::cout << " y1 (перший елемент): " << y1[0] << std::endl;
        std::cout << " y2 (перший елемент): " << y2[0] << std::endl;
        std::cout << " Y3 (перший елемент): " << Y3[0] << std::endl;
        std::cout << "--------------------------------------------" << std::endl;
        std::cout << " Фінальний результат (X): " << std::scientific << FinalResult << std::endl;
        std::cout << "============================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}