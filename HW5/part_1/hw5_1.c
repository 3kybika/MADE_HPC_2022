#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h> 
#include <string.h>

/*
Знакомство в компании
Представим, что процессы – это компания незнакомых людей, которые знакомятся с помощью следующей игры:
1) Начинает процессор 0. Случайным образом он выбирает другой процессор i и посылает ему сообщение 
со своим именем (можете случайным образом задавать имя)
2) Процессор i отсылает сообщение случайному процессору j (которые еще не участвовал в игре), в 
сообщении – все имена и ранги предыдущих процессоров в правильном порядке. Номер процессора j знает
только I, так что все должны быть начеку.
3) Игра заканчивается через N ходов. Используйте синхронную пересылку MPI_SSend
*/

/*
for run use:
/usr/bin/mpicc  hw5_1.c -o hw5_1; /usr/bin/mpirun -np 4 ./hw5_1
*/

#define NAME_LEN 21
#define ALPHABET_LEN 26


int main(int argc, char **argv) {
    int process_rank;
    int process_size;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);

    srand(process_rank * 2); // *2 - because seed for values 0 and 1 is same

    //array of process names:
    char *recieved_process_name = (char*) malloc(NAME_LEN * sizeof(char));
    char *process_name_arr = (char*) malloc(process_size * NAME_LEN * sizeof(char));
    char *process_name = (char*) malloc(NAME_LEN * sizeof(char));

    // create name of current process
    sprintf(process_name, "ps: %u, hash: ", process_rank);
    for (uint i = strlen(process_name); i < NAME_LEN - 1; ++i)
        process_name[i] = 'A' + (random() % ALPHABET_LEN);
    process_name[NAME_LEN - 1] = '\0';
    printf("[Process %u]: process name: %s\n", process_rank, process_name);

    for (uint i = 0; i < process_size; ++i)
        process_name_arr[NAME_LEN * i] = '\0';

    if(process_rank == 0) {
        // send name of 0 process
        int to_process = rand() % (process_size - 1) + 1; // get random proccess num != 0
        printf("[Process %u]: Send name from %d to %d\n", process_rank, 0, to_process);
        MPI_Ssend(process_name, NAME_LEN, MPI_UNSIGNED_CHAR, to_process, 0, MPI_COMM_WORLD);

    } else {
        // another process get name from another process
        MPI_Recv(
            &recieved_process_name[0],
            NAME_LEN,
            MPI_UNSIGNED_CHAR,
            MPI_ANY_SOURCE,
            0,
            MPI_COMM_WORLD,
            &status
        );
        printf(
            "[Process %u]: received msg: '%s' from process %d\n",
            process_rank, recieved_process_name, status.MPI_SOURCE
        );

        if(status.MPI_SOURCE != 0)
            // get process_name_arr
            MPI_Recv(&process_name_arr[0], process_size * NAME_LEN, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

        // add recieved process name to list
        for (uint i = 0; i < NAME_LEN; ++i)
            process_name_arr[status.MPI_SOURCE * NAME_LEN + i] = recieved_process_name[i];

        // add current process name to list
        for (uint i = 0; i < NAME_LEN; ++i)
            process_name_arr[process_rank * NAME_LEN + i] = process_name[i];

        // get recieved names count
        uint process_count = 0;
        for (uint i = 0; i < process_size; ++i)
            if(process_name_arr[i * NAME_LEN] != '\0')
                process_count++;
        printf("[Process %u]: received %u process names from process %d\n", process_rank, process_count, status.MPI_SOURCE);

        // print recieved arrays
        for (int i = 0; i < process_size; ++i)
        {
            if (i == process_rank) 
                printf("[Process %u] %d: %s <---current process \n", process_rank, i, process_name);
            else if (process_name_arr[i * NAME_LEN] == '\0')
                printf("[Process %u] %d:  <<Not recieved yet>>\n", process_rank, i);
            else
                printf("[Process %u] %d: %s\n", process_rank, i, process_name_arr + i * NAME_LEN * sizeof(char));
        }

        if  (process_count < process_size) {
            // send current list

            // get proccess for sending
            int to_process = rand() % (process_size - 1) + 1;
            while(process_name_arr[to_process * NAME_LEN] != '\0') {
                // like open adress
                to_process += 1;
                if (to_process >= process_size) 
                    to_process = 1;
            }

            printf("[Process %u]: sendding process name to %d\n", process_rank, to_process);
            MPI_Ssend(&process_name[0], NAME_LEN, MPI_CHAR, to_process, 0, MPI_COMM_WORLD);

            printf("[Process %u]: sendding array of names name to %d\n", process_rank, to_process);
            MPI_Ssend(&process_name_arr[0], process_size * NAME_LEN, MPI_CHAR, to_process, 0, MPI_COMM_WORLD);
        } else {
            // all names were received
            printf("\n");
            printf("============================================================\n");
            printf("[Process %u]: all process names were received\n", process_rank);

            for (uint i = 0; i < process_size; ++i) {
                printf("Proccess %d: %s \n", i, process_name_arr + i * NAME_LEN * sizeof(char));
            }
            
        }

    }

    printf("\n");

    free(recieved_process_name);
    free(process_name_arr);
    free(process_name);

    MPI_Finalize();
    return 0;
}