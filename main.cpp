#include <mpi.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>

// --- ДОПОМІЖНІ ФУНКЦІЇ ---

inline int idx(int row, int col, int num_cols) {
    return row * num_cols + col;
}

// Транспонування: (ROWS x COLS) -> (COLS x ROWS)
// Це потрібно, щоб стовпці стали рядками в пам'яті для зручної розсилки MPI
std::vector<double> transpose(const std::vector<double>& matrix, int rows, int cols) {
    std::vector<double> result(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // result[col][row] = matrix[row][col]
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // === 1. ПАРАМЕТРИ ЗАДАЧІ (580 рядків, 130 стовпців) ===
    int ROWS = 580; 
    int COLS = 130; 

    // Можливість змінити через аргументи: ./lab6 <rows> <cols>
    if (argc > 2) {
        ROWS = atoi(argv[1]);
        COLS = atoi(argv[2]);
    }

    if (world_rank == 0) {
        std::cout << "Starting Lab 6 (580x130).\n";
        std::cout << "Processes: " << world_size << "\n";
        std::cout << "Matrix: " << ROWS << " rows x " << COLS << " cols.\n";
    }

    // === 2. РОЗПОДІЛ СТОВПЦІВ (Load Balancing) ===
    // Ділимо 130 стовпців на world_size (13)
    std::vector<int> send_counts(world_size);
    std::vector<int> displs(world_size);

    int base_cols = COLS / world_size; // 130 / 13 = 10
    int remainder = COLS % world_size; // 0
    int current_disp = 0;

    for (int i = 0; i < world_size; i++) {
        send_counts[i] = base_cols + (i < remainder ? 1 : 0);
        displs[i] = current_disp;
        current_disp += send_counts[i];
    }

    int local_cols = send_counts[world_rank]; // Для 13 проц. це буде 10

    // === 3. ВИДІЛЕННЯ ПАМ'ЯТІ ===
    
    // Глобальні дані (тільки на Rank 0)
    std::vector<double> A, A1, B2, C2; 
    std::vector<double> b, b1, c1;     
    std::vector<double> full_A2; 

    // Локальні буфери
    // Ми зберігаємо стовпці транспоновано, тобто як рядки довжиною ROWS (580)
    std::vector<double> local_A_T(local_cols * ROWS);
    std::vector<double> local_A1_T(local_cols * ROWS);
    std::vector<double> local_B2_T(local_cols * ROWS);
    std::vector<double> local_C2_T(local_cols * ROWS);

    // Вектори b, b1, c1 мають розмір COLS (130), ми отримуємо шматок довжиною local_cols
    std::vector<double> local_b(local_cols);
    std::vector<double> local_b1(local_cols);
    std::vector<double> local_c1(local_cols);

    // === 4. ГЕНЕРАЦІЯ (Rank 0) ===
    if (world_rank == 0) {
        A.resize(ROWS * COLS); A1.resize(ROWS * COLS);
        B2.resize(ROWS * COLS); C2.resize(ROWS * COLS);
        full_A2.resize(ROWS * COLS);

        b.resize(COLS); b1.resize(COLS); c1.resize(COLS);

        // Заповнення даними
        for (int j = 0; j < COLS; ++j) {
            // Вектори
            b[j] = ((j + 1) % 2 == 0) ? (24.0 / ((j + 1) * (j + 1) + 4)) : 24.0;
            b1[j] = 1.0; c1[j] = 1.0;

            for (int i = 0; i < ROWS; ++i) {
                // Матриці [i][j]
                A[idx(i, j, COLS)] = 1.0;
                A1[idx(i, j, COLS)] = 1.0;
                B2[idx(i, j, COLS)] = 1.0;
                full_A2[idx(i, j, COLS)] = 1.0; 
                // Cij
                C2[idx(i, j, COLS)] = 24.0 / (i + 1 + 3.0 * (j + 1) * (j + 1));
            }
        }

        // ТРАНСПОНУВАННЯ (580x130 -> 130x580)
        // Тепер у пам'яті лежать "рядки" довжиною 580, які насправді є стовпцями
        A = transpose(A, ROWS, COLS);
        A1 = transpose(A1, ROWS, COLS);
        B2 = transpose(B2, ROWS, COLS);
        C2 = transpose(C2, ROWS, COLS);
    } else {
        full_A2.resize(ROWS * COLS);
    }

    double start_time = MPI_Wtime();

    // === 5. РОЗСИЛКА (SCATTERV) ===
    
    // Підготовка масивів для Scatterv (в одиницях double)
    std::vector<int> sc_counts(world_size), sc_displs(world_size);
    for(int i=0; i<world_size; ++i) {
        sc_counts[i] = send_counts[i] * ROWS; // Кількість елементів: 10 * 580 = 5800
        sc_displs[i] = displs[i] * ROWS;
    }

    // Розсилка стовпців (як рядків транспонованої матриці)
    MPI_Scatterv(A.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, 
                 local_A_T.data(), local_cols * ROWS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(A1.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, 
                 local_A1_T.data(), local_cols * ROWS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B2.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, 
                 local_B2_T.data(), local_cols * ROWS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(C2.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, 
                 local_C2_T.data(), local_cols * ROWS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Розсилка векторів (вони мають розмір COLS=130, ділимо просто на шматки по 10)
    MPI_Scatterv(b.data(), send_counts.data(), displs.data(), MPI_DOUBLE, 
                 local_b.data(), local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b1.data(), send_counts.data(), displs.data(), MPI_DOUBLE, 
                 local_b1.data(), local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(c1.data(), send_counts.data(), displs.data(), MPI_DOUBLE, 
                 local_c1.data(), local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast A2
    MPI_Bcast(full_A2.data(), ROWS * COLS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // === 6. ОБЧИСЛЕННЯ ===

    // --- 6.1 y1 = A * b ---
    // A [580 x 130], b [130]. y1 [580].
    // Кожен процес має 10 стовпців A (кожен висотою 580) і 10 елементів b.
    // Він рахує часткову суму вектора y1 (розміром 580).
    std::vector<double> local_y1(ROWS, 0.0);
    
    for (int k = 0; k < local_cols; ++k) {
        double val_b = local_b[k];
        for (int r = 0; r < ROWS; ++r) {
            // local_A_T[k][r] -- це A[r][global_col]
            local_y1[r] += local_A_T[idx(k, r, ROWS)] * val_b;
        }
    }
    // Сумуємо всі вектори розміром 580
    std::vector<double> y1(ROWS);
    MPI_Allreduce(local_y1.data(), y1.data(), ROWS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // --- 6.2 y2 = A1 * (b1 - 24*c1) ---
    std::vector<double> local_y2(ROWS, 0.0);
    for (int k = 0; k < local_cols; ++k) {
        double val_vec = local_b1[k] - (24.0 * local_c1[k]);
        for (int r = 0; r < ROWS; ++r) {
            local_y2[r] += local_A1_T[idx(k, r, ROWS)] * val_vec;
        }
    }
    std::vector<double> y2(ROWS);
    MPI_Allreduce(local_y2.data(), y2.data(), ROWS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // --- 6.3 Y3 (поелементно для прикладу) ---
    // Y3 = A2 * (B2 + 24*C2) -> Поелементно, бо множення матриць тут неможливе (розмірності)
    std::vector<double> local_Y3_T(local_cols * ROWS);

    for (int k = 0; k < local_cols; ++k) {
        for (int r = 0; r < ROWS; ++r) {
            double val_B = local_B2_T[idx(k, r, ROWS)];
            double val_C = local_C2_T[idx(k, r, ROWS)];
            
            // Глобальний індекс стовпця
            int global_col = displs[world_rank] + k;
            double val_A = full_A2[idx(r, global_col, COLS)];

            local_Y3_T[idx(k, r, ROWS)] = val_A * (val_B + 24.0 * val_C);
        }
    }

    // === 7. ЗБІР РЕЗУЛЬТАТІВ (GATHERV) ===
    std::vector<double> Y3_T;
    if (world_rank == 0) Y3_T.resize(ROWS * COLS);

    // Збираємо транспоновані шматки (по 5800 елементів)
    MPI_Gatherv(local_Y3_T.data(), local_cols * ROWS, MPI_DOUBLE,
                Y3_T.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // === 8. ФІНАЛ (Rank 0) ===
    if (world_rank == 0) {
        // Транспонуємо Y3 назад у вигляд 580x130
        std::vector<double> Y3 = transpose(Y3_T, COLS, ROWS);
        
        // Скалярний добуток y1 * y2 (розмір 580)
        double y1_dot_y2 = 0.0;
        for(int i=0; i<ROWS; ++i) y1_dot_y2 += y1[i] * y2[i];

        double Y3_sum = 0.0;
        for(auto v : Y3) Y3_sum += v;

        double end_time = MPI_Wtime();

        std::cout << "Done. Time: " << (end_time - start_time) * 1000.0 << " ms.\n";
        std::cout << "Debug Info: y1*y2=" << y1_dot_y2 << ", Sum(Y3)=" << Y3_sum << std::endl;
    }

    MPI_Finalize();
    return 0;
}