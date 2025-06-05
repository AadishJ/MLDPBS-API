#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <string.h>

#define TOTAL_RAM_SIZE (512 * 1024 * 1024) // 512 MB
#define MAX_BLOCK_TYPES 10                 // Maximum number of different block sizes to support
#define ITERATION_COUNT 100000             // Total number of iterations for testing
#define NUM_THREADS 8                      // Number of threads

typedef struct
{
    uint8_t *pool;
    int pool_size;
    bool *free_blocks;
    int block_size;
    pthread_spinlock_t lock; // Spin-lock for synchronization
} Partition;

typedef struct
{
    Partition partition;
} BuddyAllocator;

typedef struct
{
    BuddyAllocator *allocator;
    double *total_times;
    int *counts;
    int *block_sizes;
    int block_size_count;
} ThreadData;

void init(BuddyAllocator *allocator, int default_block_size);
void *allocateBuddySystem(BuddyAllocator *allocator, int size);
void *allocateBlock(Partition *partition, int size);
void deallocate(BuddyAllocator *allocator, void *ptr);
void cleanup(BuddyAllocator *allocator);
void *thread_worker(void *args);
int read_data_from_file(const char *filepath, double *percentages, int *block_sizes, int max_count);

void init(BuddyAllocator *allocator, int default_block_size)
{
    allocator->partition.pool_size = TOTAL_RAM_SIZE;
    allocator->partition.block_size = default_block_size; // Use provided block size
    allocator->partition.pool = (uint8_t *)malloc(TOTAL_RAM_SIZE);
    allocator->partition.free_blocks = (bool *)malloc(TOTAL_RAM_SIZE / default_block_size * sizeof(bool));
    pthread_spin_init(&allocator->partition.lock, 0);

    for (int j = 0; j < TOTAL_RAM_SIZE / default_block_size; ++j)
    {
        allocator->partition.free_blocks[j] = true;
    }
}

void *allocateBuddySystem(BuddyAllocator *allocator, int size)
{
    pthread_spin_lock(&allocator->partition.lock);
    void *ptr = allocateBlock(&allocator->partition, size);
    pthread_spin_unlock(&allocator->partition.lock);
    return ptr;
}

void *allocateBlock(Partition *partition, int size)
{
    int block_count = partition->pool_size / partition->block_size;
    for (int j = 0; j < block_count; ++j)
    {
        if (partition->free_blocks[j] && partition->block_size >= size)
        {
            partition->free_blocks[j] = false;
            return partition->pool + j * partition->block_size;
        }
    }
    return NULL;
}

void deallocate(BuddyAllocator *allocator, void *ptr)
{
    if (ptr >= (void *)allocator->partition.pool && ptr < (void *)(allocator->partition.pool + allocator->partition.pool_size))
    {
        int block_index = ((uint8_t *)ptr - allocator->partition.pool) / allocator->partition.block_size;
        allocator->partition.free_blocks[block_index] = true;
    }
}

void cleanup(BuddyAllocator *allocator)
{
    free(allocator->partition.pool);
    free(allocator->partition.free_blocks);
    pthread_spin_destroy(&allocator->partition.lock);
}

void *thread_worker(void *args)
{
    ThreadData *thread_data = (ThreadData *)args;
    BuddyAllocator *allocator = thread_data->allocator;
    int block_size_count = thread_data->block_size_count;
    int *block_sizes = thread_data->block_sizes;
    int iterations_per_thread = ITERATION_COUNT / NUM_THREADS;

    for (int b = 0; b < block_size_count; ++b)
    {
        clock_t start, end;
        double total_time = 0.0;
        start = clock();
        for (int i = 0; i < iterations_per_thread; ++i)
        {
            void *ptr = allocateBuddySystem(allocator, block_sizes[b]);
            if (ptr != NULL)
            {
                deallocate(allocator, ptr);
            }
        }
        end = clock();
        total_time += ((double)(end - start) / CLOCKS_PER_SEC) * 1000;
        thread_data->total_times[b] += total_time;
        thread_data->counts[b] += 1;
    }
    return NULL;
}

// Read percentages and block sizes from file
int read_data_from_file(const char *filepath, double *percentages, int *block_sizes, int max_count)
{
    FILE *file = fopen(filepath, "r");
    if (file == NULL)
    {
        printf("Error: Cannot open file %s\n", filepath);
        return 0;
    }

    char line[256];
    int percentage_count = 0;
    int block_size_count = 0;
    int reading_percentages = 1; // Flag to track what we're reading

    // First pass: count how many lines we have
    int total_lines = 0;
    while (fgets(line, sizeof(line), file))
    {
        line[strcspn(line, "\n")] = 0; // Remove newline
        if (strlen(line) > 0)          // Skip empty lines
        {
            total_lines++;
        }
    }

    // Reset file pointer
    rewind(file);

    // Assume first half are percentages, second half are block sizes
    int expected_count = total_lines / 2;
    int current_line = 0;

    while (fgets(line, sizeof(line), file))
    {
        // Remove trailing newline
        line[strcspn(line, "\n")] = 0;

        // Skip empty lines
        if (strlen(line) == 0)
        {
            continue;
        }

        if (current_line < expected_count)
        {
            // Reading percentages
            char *endptr;
            double percentage = strtod(line, &endptr);
            if (endptr != line && percentage_count < max_count)
            {
                percentages[percentage_count] = percentage;
                percentage_count++;
            }
        }
        else
        {
            // Reading block sizes
            char *endptr;
            int block_size = (int)strtol(line, &endptr, 10);
            if (endptr != line && block_size_count < max_count)
            {
                block_sizes[block_size_count] = block_size;
                block_size_count++;
            }
        }
        current_line++;
    }

    fclose(file);

    // Return the minimum count to ensure we have matching pairs
    int min_count = (percentage_count < block_size_count) ? percentage_count : block_size_count;
    return min_count;
}

int main()
{
    // Path to percentages file
    char file_path[512];

    // Try to find the file in different possible locations
    const char *possible_paths[] = {
        "predictions/percentages.txt",
    };

    int path_found = 0;
    for (int i = 0; i < sizeof(possible_paths) / sizeof(possible_paths[0]); i++)
    {
        FILE *test = fopen(possible_paths[i], "r");
        if (test != NULL)
        {
            fclose(test);
            strcpy(file_path, possible_paths[i]);
            path_found = 1;
            break;
        }
    }

    if (!path_found)
    {
        printf("Error: percentages.txt not found in any expected location\n");
        printf("Using default values\n");

        // Use default values
        double default_percentages[] = {40.0, 35.0, 25.0};
        int default_block_sizes[] = {482, 360, 6};
        int block_size_count = 3;

        BuddyAllocator allocator;
        init(&allocator, 482); // Initialize with the largest block size

        pthread_t threads[NUM_THREADS];
        ThreadData thread_data[NUM_THREADS];
        double total_times[MAX_BLOCK_TYPES] = {0};
        int counts[MAX_BLOCK_TYPES] = {0};

        for (int i = 0; i < NUM_THREADS; ++i)
        {
            thread_data[i].allocator = &allocator;
            thread_data[i].total_times = total_times;
            thread_data[i].counts = counts;
            thread_data[i].block_sizes = default_block_sizes;
            thread_data[i].block_size_count = block_size_count;
            pthread_create(&threads[i], NULL, thread_worker, &thread_data[i]);
        }

        for (int i = 0; i < NUM_THREADS; ++i)
        {
            pthread_join(threads[i], NULL);
        }

        printf("Results:\n");
        printf("Block Size\tAvg Time Taken (ms)\n");
        for (int b = 0; b < block_size_count; ++b)
        {
            double avg_time = total_times[b] / counts[b];
            printf("%d\t%.2f\n", default_block_sizes[b], avg_time);
        }

        cleanup(&allocator);
        return 0;
    }

    // Read data from file
    double percentages[MAX_BLOCK_TYPES];
    int block_sizes[MAX_BLOCK_TYPES];
    int count = read_data_from_file(file_path, percentages, block_sizes, MAX_BLOCK_TYPES);

    if (count == 0)
    {
        printf("No valid data found in file. Exiting.\n");
        return 1;
    }

    printf("Found %d entries in file\n", count);
    for (int i = 0; i < count; i++)
    {
        printf("Entry %d: %.2f%% - Block size: %d bytes\n", i + 1, percentages[i], block_sizes[i]);
    }

    // Initialize the allocator with the largest block size for simplicity
    int largest_block = block_sizes[0];
    for (int i = 1; i < count; i++)
    {
        if (block_sizes[i] > largest_block)
        {
            largest_block = block_sizes[i];
        }
    }

    BuddyAllocator allocator;
    init(&allocator, largest_block);

    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    double total_times[MAX_BLOCK_TYPES] = {0};
    int counts[MAX_BLOCK_TYPES] = {0};

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        thread_data[i].allocator = &allocator;
        thread_data[i].total_times = total_times;
        thread_data[i].counts = counts;
        thread_data[i].block_sizes = block_sizes;
        thread_data[i].block_size_count = count;
        pthread_create(&threads[i], NULL, thread_worker, &thread_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        pthread_join(threads[i], NULL);
    }

    printf("\nResults:\n");
    printf("Block Size\tAvg Time Taken (ms)\n");
    for (int b = 0; b < count; ++b)
    {
        double avg_time = total_times[b] / counts[b];
        printf("%d\t%.2f\n", block_sizes[b], avg_time);
    }

    cleanup(&allocator);
    return 0;
}