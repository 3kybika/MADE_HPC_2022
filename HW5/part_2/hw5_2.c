#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h> 
#include <string.h>
#include <stdbool.h>
#include <unistd.h>

/*
Параллельный одномерный клеточный автомат (периодические граничные условия).
С помощью MPI распараллельте одномерный клеточный автомат Вольфрама (Rule110).
Игра происходит следующим образом:
1) Инициализируйте одномерный массив 0 и 1 случайным образом
2) В зависимости от значений: левого соседа, себя, правого соседа на следующем шаге клетка либо меняет значение, либо остается той же. Посмотрите, например, что значит Rule110 (https://en.wikipedia.org/wiki/Rule_110)
Сделайте периодические и непериодические граничные условия (5 баллов)
Работает параллельный код на нескольких процессах (20 баллов)
Имплементированы клетки-призраки (ghost cells) (10 балла)
Можно поменять правило игры (сделать одно из 256) (20 баллов)
График ускорения работы программы от кол-ва процессов (5 баллов)
Картинка эволюции для одного правила (15 баллов)

/*
for run use:
/usr/bin/mpicc  hw5_2.c -o hw5_2; /usr/bin/mpirun -np 4 ./hw5_2 <rule>

Args:
-ht, --height: int - height of rendered area (iterations num), default 50
-w, --width: int - width of rendered area 60
-r, --rule: int - rule of cellular automaton 110
-s, --stat_only: flag - print only time without rendering picture
*/

#define CANVAS_SIZE 60
#define ALPHABET_LEN 26
#define DEFAULT_RULE 110
#define STATES_NUM 8
#define ITERATION_NUM 50
#define STAT_ONLY false

void print_canvas(int *canvas, int canvas_size) {
    for (int i = 1; i <= canvas_size; ++i) {
        if(canvas[i])
            printf(" ");
        else
            printf("▉");
    }
}

int main(int argc, char **argv) {

    int process_rank;
    int process_size;

    const int PROCESS_TAG = 1;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);

    // parse args
    int iterations_num = ITERATION_NUM; // --height -h
    int canvas_size = CANVAS_SIZE;      // --width -w
    int rule = DEFAULT_RULE;            // --rule -r
    bool stat_only = STAT_ONLY;         // --stat_only -s

    for (int optind = 1; optind < argc; optind++) {
        if (argv[optind][0] != '-')
            continue;

        if (strcmp(argv[optind], "-ht") == 0 || strcmp(argv[optind], "--height") == 0) {
            iterations_num = atoi((char*) argv[optind+1]);
            continue;
        }

        if (strcmp(argv[optind], "-w") == 0 || strcmp(argv[optind], "--width") == 0) {
            canvas_size = atoi((char*) argv[optind+1]);
            continue;
        }

        if (strcmp(argv[optind], "-r") == 0 || strcmp(argv[optind], "--rule") == 0) {
            rule = atoi((char*) argv[optind+1]);
            continue;
        }

        if (strcmp(argv[optind], "-s") == 0 || strcmp(argv[optind], "--stat_only") == 0) {
            stat_only = 1;
            continue;
        }

        if (strcmp(argv[optind], "-h") == 0 || strcmp(argv[optind], "--help") == 0) {
            if (process_rank == 0) {
                printf("Args:\n");
                printf("-ht, --height: int - height of rendered area (iterations num), default %d\n", ITERATION_NUM);
                printf("-w, --width: int - width of rendered area %d\n", CANVAS_SIZE);
                printf("-r, --rule: int - rule of cellular automaton %d\n", DEFAULT_RULE);
                printf("-s, --stat_only: flag - print only time without rendering picture\n");
            }
            MPI_Finalize();
            return 0;
        }
    }

    // create "tube" map: each procvess will create it's own area on this map
    int left_neighbour = process_rank - 1;
    if (left_neighbour == -1)
        left_neighbour = process_size - 1;

    int right_neighbour = process_rank + 1;
    if (right_neighbour == process_size)
        right_neighbour = 0;

    // get chunk size
    int chunk_size = (int)(canvas_size / process_size);
    if (!stat_only && process_rank == 0 && canvas_size % process_size != 0)
        printf(
            "Warning: Canvas size %d is not integer divisible by the number of processes %d; The size, that will be rendered is %d\n",
            canvas_size, process_size, chunk_size * process_size
        );
    
    // init map
    int *canvas = (int*) malloc((chunk_size + 2) * sizeof(int));
    int *tmp_canvas = (int*) malloc((chunk_size + 2) * sizeof(int));

    srand(process_rank * 2);
    //srand(time(NULL));
    for (int i = 1; i <= chunk_size; ++i)
        canvas[i] = rand() % 2;

    // init rule
    int rule_arr[STATES_NUM];
    for(int i = 0; i < STATES_NUM; ++i){
        rule_arr[i] = rule & 1;
        rule >>= 1;
    }

    double time_elapsed = MPI_Wtime();

    for(int i = 0; i < iterations_num; ++i) {
        // send ghost cells - ghost has shift = 1 
        MPI_Send(&canvas[1], 1, MPI_INT, left_neighbour, PROCESS_TAG, MPI_COMM_WORLD);
        MPI_Send(&canvas[chunk_size], 1, MPI_INT, right_neighbour, PROCESS_TAG, MPI_COMM_WORLD);

        // receive ghost cells
        MPI_Recv(&canvas[0], 1, MPI_INT, left_neighbour, PROCESS_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&canvas[chunk_size + 1], 1, MPI_INT, right_neighbour, PROCESS_TAG, MPI_COMM_WORLD, &status);

        // cellular automaton
        for(int i = 1; i <= chunk_size; ++i) {
            int cur_state = (canvas[i - 1] << 2) + (canvas[i] << 1) + canvas[i + 1];
            //printf("r%d", cur_state);
            tmp_canvas[i] = rule_arr[cur_state];
        }

        for(int i =1; i <= chunk_size; ++i)
            canvas[i] = tmp_canvas[i];

        // agg results
        if (process_rank == 0) {
            if (!stat_only) print_canvas(canvas, chunk_size);
            
            for(int proc_num = 1; proc_num < process_size; ++ proc_num) {
                MPI_Recv(&tmp_canvas[1], chunk_size, MPI_INT, proc_num, PROCESS_TAG, MPI_COMM_WORLD, &status);
                if (!stat_only) print_canvas(tmp_canvas, chunk_size);
            }
                if (!stat_only) printf("\n");
        } else {
            MPI_Send(&canvas[1], chunk_size, MPI_INT, 0, PROCESS_TAG,  MPI_COMM_WORLD);
        }
    }

    time_elapsed = MPI_Wtime() - time_elapsed;
    if (process_rank == 0)
        if (!stat_only)
            printf("elapsed time: %f\n", time_elapsed);
        else 
            printf("%f\n", time_elapsed);

    free(canvas);
    free(tmp_canvas);

    MPI_Finalize();

    return 0;
} 