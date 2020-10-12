#include "read_file.h"
#include <random>

int main(int argc, char **argv){
    //Generate random generator
    double *pos_x = (double*) malloc(sizeof(double) * 100000);
    double *pos_y = (double*) malloc(sizeof(double) * 100000);
    double *pos_z = (double*) malloc(sizeof(double) * 100000);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for(int pir = 0; pir < 100000; pir++){
      double x_pos = dis(gen);
      double y_pos = dis(gen);
      double z_pos = dis(gen);
      pos_x[pir] = x_pos;
      pos_y[pir] = y_pos;
      pos_z[pir] = z_pos;
    }
    write_position(pos_x, pos_y, pos_z, 100000);
    read_positions(pos_x, pos_y, pos_z);
	free(pos_x);
	free(pos_y);
	free(pos_z);
}
