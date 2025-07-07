#include <cmath>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <ncurses.h>
#include <unistd.h>
#include <vector>

constexpr long long int NUMBER_OF_SAMPLES = 1000000000;
constexpr int GRID_SIZE = 50;
constexpr double ASPECT_RATIO = 2.0; // aspect of terminal height:width

int world_size = 0;

void draw_progress_bars(const std::vector<int>& progress_per_rank,
                        int world_size) {
    clear();
    printw("Number of Cores: %2d, Total Number of samples: %2lld", world_size,
           NUMBER_OF_SAMPLES);

    for (int i = 0; i < world_size; ++i) {
        int progress = progress_per_rank[i];
        mvprintw(i + 1, 0, "Rank %2d: [", i);
        int bar_width = 30;
        int pos = (progress * bar_width) / 100;
        for (int j = 0; j < bar_width; ++j) {
            if (j < pos)
                printw("#");
            else
                printw(" ");
        }
        printw("] %3d%%", progress);
    }
}

void draw_quarter_circle() {

    for (int y = 0; y < GRID_SIZE; ++y) {
        for (int x = 0; x < GRID_SIZE; ++x) {
            double xf = (double)x / GRID_SIZE;
            double yf = (double)(GRID_SIZE - y) / GRID_SIZE;

            if (xf * xf + yf * yf <= 1.0) {
                mvaddch(world_size + 2 + y / ASPECT_RATIO, x, '.');
            }
        }
    }

    for (int i = 0; i < GRID_SIZE; ++i) {
        mvaddch(world_size + 2 + (i / ASPECT_RATIO), 0, '|');
        mvaddch(world_size + 2 + (GRID_SIZE / ASPECT_RATIO), i, '-');
    }
    mvaddch(world_size + 2 + (GRID_SIZE / ASPECT_RATIO), 0, '+');
}

void draw_point_plot(const std::vector<std::pair<double, double>>& points) {
    draw_quarter_circle();
    for (const auto& p : points) {
        int x = static_cast<int>(p.first * GRID_SIZE);
        int y = static_cast<int>(p.second * GRID_SIZE / ASPECT_RATIO);
        if (x >= GRID_SIZE)
            x = GRID_SIZE - 1;
        if (y >= GRID_SIZE)
            y = GRID_SIZE - 1;

        double distance = p.first * p.first + p.second * p.second;
        if (distance <= 1.0) {
            attron(COLOR_PAIR(1));
            mvaddch((GRID_SIZE / ASPECT_RATIO - y) + world_size + 2, x, 'o');
            attroff(COLOR_PAIR(1));
        } else {
            attron(COLOR_PAIR(2));
            mvaddch((GRID_SIZE / ASPECT_RATIO - y) + world_size + 2, x, 'x');
            attroff(COLOR_PAIR(2));
        }
    }
    refresh();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    double start_time = MPI_Wtime();

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const long long int total_points = NUMBER_OF_SAMPLES;
    long long int points_per_proc = total_points / world_size;

    srand(time(NULL) + world_rank);

    int progress = 0;
    int last_reported = -1;

    long long int local_count = 0;

    std::vector<int> progress_data;
    std::vector<std::pair<double, double>> point_data;

    if (world_rank == 0) {
        initscr();
        start_color();
        init_pair(1, COLOR_GREEN, COLOR_BLACK); // Inside point
        init_pair(2, COLOR_RED, COLOR_BLACK);   // Outside point
        noecho();
        cbreak();
        curs_set(0);
        progress_data.resize(world_size, 0);
        point_data.resize(world_size);
    }

    std::pair<double, double> latest_point = {0.0, 0.0};

    for (long long int i = 0; i < points_per_proc; ++i) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        latest_point = {x, y};

        if (x * x + y * y <= 1.0)
            local_count++;

        progress = (100 * i) / points_per_proc;

        if (progress != last_reported) {
            last_reported = progress;

            if (world_rank == 0) {
                MPI_Gather(MPI_IN_PLACE, 1, MPI_INT, progress_data.data(), 1,
                           MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Gather(MPI_IN_PLACE, 2, MPI_DOUBLE, point_data.data(), 2,
                           MPI_DOUBLE, 0, MPI_COMM_WORLD);
                progress_data[0] = progress;
                point_data[0] = {latest_point.first, latest_point.second};
                draw_progress_bars(progress_data, world_size);
                draw_point_plot(point_data);
            } else {
                MPI_Gather(&progress, 1, MPI_INT, nullptr, 0, MPI_INT, 0,
                           MPI_COMM_WORLD);
                double send_buf[2] = {latest_point.first, latest_point.second};
                MPI_Gather(send_buf, 2, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0,
                           MPI_COMM_WORLD);
            }
        }
    }

    long long int global_count = 0;
    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG_LONG_INT, MPI_SUM, 0,
               MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    double duration = end_time - start_time;

    if (world_rank == 0) {
        std::fill(progress_data.begin(), progress_data.end(), 100);
        draw_progress_bars(progress_data, world_size);
        draw_point_plot(point_data);

        double pi = 4.0 * global_count / total_points;
        mvprintw(world_size + 5 + (GRID_SIZE / ASPECT_RATIO) + 2, 0,
                 "Approximated Pi: %.10f", pi);
        mvprintw(world_size + 5 + (GRID_SIZE / ASPECT_RATIO) + 3, 0,
                 "Execution Time:  %.4f seconds", duration);
        mvprintw(world_size + 5 + (GRID_SIZE / ASPECT_RATIO) + 5, 0,
                 "Press any key to exit...");
        refresh();
        getch();
        endwin();
    }

    MPI_Finalize();
    return 0;
}
