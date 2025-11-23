#include <mpi.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>

// Допоміжна функція для індексації
inline int idx(int row, int col, int N) {
    return row * N + col;
}

// Транспонування (A -> A^T) для зручної відправки стовпців
std::vector<double> transpose(const std::vector<double>& matrix, int N) {
    std::vector<double> result(N * N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            result[idx(j, i, N)] = matrix[idx(i, j, N)];
        }
    }
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Зчитуємо N з аргументів (за замовчуванням 130)
    int N = 130; 
    if (argc > 1) N = atoi(argv[1]);

    if (world_rank == 0) {
        std::cout << "Starting Lab 6. Variant 23.\n";
        std::cout << "Processes: " << world_size << ", Matrix N: " << N << "\n";
    }

    // === РОЗРАХУНОК РОЗПОДІЛУ НАВАНТАЖЕННЯ (Load Balancing) ===
    // Визначаємо, скільки стовпців дістанеться кожному процесору
    std::vector<int> send_counts(world_size); // Кількість стовпців для кожного рангу
    std::vector<int> displs(world_size);      // Зміщення (індекс початку)
    
    int base_cols = N / world_size;
    int remainder = N % world_size;
    int current_disp = 0;

    for (int i = 0; i < world_size; i++) {
        // Якщо є залишок, перші процеси отримують на 1 стовпець більше
        send_counts[i] = base_cols + (i < remainder ? 1 : 0);
        displs[i] = current_disp;
        current_disp += send_counts[i];
    }

    // Локальна кількість стовпців для поточного процесу
    int local_cols = send_counts[world_rank];

    // === ПІДГОТОВКА БУФЕРІВ ===
    // Повні матриці (тільки на Rank 0)
    std::vector<double> A, A1, A2, B2, C2; 
    std::vector<double> b, b1, c1;         
    
    // Локальні буфери
    // local_cols стовпців, кожен висотою N. Зберігаємо транспоновано (як рядки).
    std::vector<double> local_A_T(local_cols * N); 
    std::vector<double> local_A1_T(local_cols * N);
    std::vector<double> local_B2_T(local_cols * N);
    std::vector<double> local_C2_T(local_cols * N);
    
    std::vector<double> full_A2(N * N); // A2 потрібна всім повністю
    
    std::vector<double> local_b(local_cols);
    std::vector<double> local_b1(local_cols);
    std::vector<double> local_c1(local_cols);

    // === 1. ГЕНЕРАЦІЯ (Rank 0) ===
    if (world_rank == 0) {
        A.resize(N * N); A1.resize(N * N); A2.resize(N * N);
        B2.resize(N * N); C2.resize(N * N);
        b.resize(N); b1.resize(N); c1.resize(N);

        for (int i = 0; i < N; ++i) {
            b[i] = ((i + 1) % 2 == 0) ? (24.0 / ((i + 1) * (i + 1) + 4)) : 24.0;
            b1[i] = 1.0; c1[i] = 1.0; 
            for (int j = 0; j < N; ++j) {
                A[idx(i, j, N)] = 1.0; A1[idx(i, j, N)] = 1.0;
                A2[idx(i, j, N)] = 1.0; B2[idx(i, j, N)] = 1.0;
                C2[idx(i, j, N)] = 24.0 / (i + 1 + 3 * (j + 1) * (j + 1));
            }
        }
        full_A2 = A2; 
        
        // Транспонуємо для розсилки стовпців
        A = transpose(A, N); A1 = transpose(A1, N);
        B2 = transpose(B2, N); C2 = transpose(C2, N);
    }

    double start_time = MPI_Wtime();

    // === 2. РОЗСИЛКА (SCATTERV) ===
    // Для Scatterv нам потрібні масиви кількості елементів (а не стовпців)
    std::vector<int> sc_counts(world_size), sc_displs(world_size);
    for(int i=0; i<world_size; ++i) {
        sc_counts[i] = send_counts[i] * N; // Кількість double
        sc_displs[i] = displs[i] * N;      // Зміщення в double
    }

    // Розсилаємо матриці
    MPI_Scatterv(A.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, 
                 local_A_T.data(), local_cols * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(A1.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, 
                 local_A1_T.data(), local_cols * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B2.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, 
                 local_B2_T.data(), local_cols * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(C2.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE, 
                 local_C2_T.data(), local_cols * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Розсилаємо вектори (тут використовуємо send_counts, бо вектори 1D)
    MPI_Scatterv(b.data(), send_counts.data(), displs.data(), MPI_DOUBLE, 
                 local_b.data(), local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b1.data(), send_counts.data(), displs.data(), MPI_DOUBLE, 
                 local_b1.data(), local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(c1.data(), send_counts.data(), displs.data(), MPI_DOUBLE, 
                 local_c1.data(), local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // A2 всім
    MPI_Bcast(full_A2.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // === 3. ОБЧИСЛЕННЯ ===
    
    // y1 = A * b
    std::vector<double> local_y1_sum(N, 0.0);
    for (int k = 0; k < local_cols; ++k) {
        double val = local_b[k];
        for (int i = 0; i < N; ++i) local_y1_sum[i] += local_A_T[idx(k, i, N)] * val;
    }
    std::vector<double> y1(N);
    MPI_Allreduce(local_y1_sum.data(), y1.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // y2 = A1 * (b1 - 24*c1)
    std::vector<double> local_y2_sum(N, 0.0);
    for (int k = 0; k < local_cols; ++k) {
        double factor = local_b1[k] - (local_c1[k] * 24.0);
        for (int i = 0; i < N; ++i) local_y2_sum[i] += local_A1_T[idx(k, i, N)] * factor;
    }
    std::vector<double> y2(N);
    MPI_Allreduce(local_y2_sum.data(), y2.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Y3 = A2 * (B2 + 24*C2)
    std::vector<double> local_Y3_cols_T(local_cols * N);
    std::vector<double> temp_col(N);

    for (int k = 0; k < local_cols; ++k) {
        // Формуємо стовпець (B2 + 24*C2)
        for(int r = 0; r < N; ++r) {
            temp_col[r] = local_B2_T[idx(k, r, N)] + 24.0 * local_C2_T[idx(k, r, N)];
        }
        // Множимо A2 на цей стовпець
        for (int i = 0; i < N; ++i) {
            double dot = 0.0;
            for (int j = 0; j < N; ++j) dot += full_A2[idx(i, j, N)] * temp_col[j];
            local_Y3_cols_T[idx(k, i, N)] = dot;
        }
    }

    // === 4. ЗБІР РЕЗУЛЬТАТІВ (GATHERV) ===
    std::vector<double> Y3_T;
    if (world_rank == 0) Y3_T.resize(N * N);

    MPI_Gatherv(local_Y3_cols_T.data(), local_cols * N, MPI_DOUBLE,
                Y3_T.data(), sc_counts.data(), sc_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // === 5. ФІНАЛ ===
    if (world_rank == 0) {
        // Транспонуємо Y3 назад
        std::vector<double> Y3 = transpose(Y3_T, N);
        
        // Фінальні скалярні обрахунки
        double y1_dot_y2 = 0.0;
        for(int i=0; i<N; ++i) y1_dot_y2 += y1[i] * y2[i];

        // Сума Y3 (для прикладу)
        double Y3_sum = 0.0;
        for(auto v : Y3) Y3_sum += v;

        double end_time = MPI_Wtime();
        
        std::cout << "Done. Time: " << (end_time - start_time) * 1000.0 << " ms.\n";
        std::cout << "Debug Info: y1*y2=" << y1_dot_y2 << ", Sum(Y3)=" << Y3_sum << std::endl;
    }

    MPI_Finalize();
    return 0;
}