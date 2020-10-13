#include "read_file.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

#define N_PARTS 100000
//We're going to use a cutoff of 0.1 and dimensions of 1x1x1 for easyness
#define CUTOFF 0.1
#define CUTOFF2 0.01
#define MASS 1
#define XDIM 1.0
#define YDIM 1.0
#define ZDIM 1.0
#define TIMESTEP 0.01

#define TRADEQUEUE_SIZE 100

#define TRUE 1
#define FALSE 0

struct part{
  double pos_x;
  double pos_y;
  double pos_z;
  double vel_x;
  double vel_y;
  double vel_z;
  double acc_x;
  double acc_y;
  double acc_z;
  int _valid;
  int64_t cell_id;
};

struct cell {
    struct part* parts;
    int id;
    int nparts;
};

int64_t get_cell_1d(int x_cell, int y_cell, int z_cell){

    //Equation to use:
    // z_cell * 100 + y_cell *10 + x_cell
    // Relies on being a 10x10x10 grid which we set up with the defines
    return z_cell * 100 + y_cell * 10 + x_cell;
}

int64_t get_cell(double x_pos, double y_pos, double z_pos){
  //Compute x/y/z cell.
  int x_cell = floor(x_pos / CUTOFF);
  int y_cell = floor(y_pos / CUTOFF);
  int z_cell = floor(z_pos / CUTOFF);
  return get_cell_1d(x_cell, y_cell, z_cell);
}

int64_t get_next_cell(){
    static int64_t cell = 0;
    static int count = 0;
    if(count == TRADEQUEUE_SIZE){
        count = 1;
        cell = cell + 1;
    }else{
        count++;
    }
    return cell;
}

void delete_cell(struct cell* cell){
  free(cell->parts);
}

void init_cell(struct cell* cell, struct part* parts, int size_parts){
  int count = 0; //TRADEQUEUE_SIZE;
  for(int i = 0; i < size_parts; i++){
    if(parts[i].cell_id == cell->id){
      count++;
    }
  }
  cell->parts = (struct part*) malloc(sizeof(struct part) * count);
  cell->nparts = count;
  int counter = 0;
  for(int i = 0; i < size_parts; i++){
    if(parts[i].cell_id == cell->id){
        memcpy(&cell->parts[counter], &parts[i], sizeof(struct part));
        counter++;
    }
  }
}

void self_task(struct cell* cell){
  for(int i = 0; i < cell->nparts; i++){
    if(!cell->parts[i]._valid){
        continue;
    }
    double p1_x = cell->parts[i].pos_x;
    double p1_y = cell->parts[i].pos_y;
    double p1_z = cell->parts[i].pos_z;
    double fix = 0., fiy = 0., fiz = 0.;
    for(int j = i+1; j < cell->nparts; j++){
        if(!cell->parts[j]._valid){
            continue;
        }
        double p2_x = cell->parts[j].pos_x;
        double p2_y = cell->parts[j].pos_y;
        double p2_z = cell->parts[j].pos_z;

        double dx = p1_x - p2_x;
        double dy = p1_y - p2_y;
        double dz = p1_z - p2_z;

        double r2 = dx * dx + dy * dy + dz * dz;
        //If within cutoff
        if(r2 <= CUTOFF2){
          //Do some computation
          double r = sqrtf(r2);
          double ir2 = 1.0 / r2;
          double sig_r = 0.1 / r;
          double sig_r2 = sig_r * sig_r;
          double sig_r4 = sig_r2 * sig_r2;
          double sig_r6 = sig_r4 * sig_r2;
          double sig_r12 = sig_r6 * sig_r6;
          double gamma = 24 * (2.0*sig_r12 - sig_r6) * ir2;

          double fx = gamma * dx;
          double fy = gamma * dy;
          double fz = gamma * dz;

          fix = fix + fx;
          fiy = fiy + fy;
          fiz = fiz + fz;
          cell->parts[j].acc_x -= fx;
          cell->parts[j].acc_y -= fy;
          cell->parts[j].acc_z -= fz;
//          printf("Found a neighbour\n");
        }
    }
    cell->parts[i].acc_x += fix;
    cell->parts[i].acc_y += fiy;
    cell->parts[i].acc_z += fiz;
  }
}

void timestep_task(struct cell* cell){
    for(int i = 0; i < cell->nparts; i++){
        if(cell->parts[i]._valid){
            cell->parts[i].pos_x = cell->parts[i].pos_x + cell->parts[i].vel_x*TIMESTEP;
            cell->parts[i].pos_y = cell->parts[i].pos_y + cell->parts[i].vel_y*TIMESTEP;
            cell->parts[i].pos_z = cell->parts[i].pos_z + cell->parts[i].vel_z*TIMESTEP;
            cell->parts[i].vel_x = cell->parts[i].vel_x + cell->parts[i].acc_x*TIMESTEP;
            cell->parts[i].vel_y = cell->parts[i].vel_y + cell->parts[i].acc_y*TIMESTEP;
            cell->parts[i].vel_z = cell->parts[i].vel_z + cell->parts[i].acc_z*TIMESTEP;
        }
    }
}

void init_parts(struct part* parts, int size_parts){
  double *pos_x = (double*) malloc(sizeof(double) * N_PARTS);
  double *pos_y = (double*) malloc(sizeof(double) * N_PARTS);
  double *pos_z = (double*) malloc(sizeof(double) * N_PARTS);
  read_positions(pos_x, pos_y, pos_z);

  for(int i = 0; i < N_PARTS; i++){
    parts[i].pos_x = pos_x[i];
    parts[i].pos_y = pos_y[i];
    parts[i].pos_z = pos_z[i];
    parts[i].vel_x = 0.0;
    parts[i].vel_y = 0.0;
    parts[i].vel_z = 0.0;
    parts[i].acc_x = 0.0;
    parts[i].acc_y = 0.0;
    parts[i].acc_z = 0.0;
    parts[i]._valid = TRUE;
    parts[i].cell_id = get_cell(pos_x[i], pos_y[i], pos_z[i]);
  }
  for(int i = N_PARTS; i < size_parts; i++){
    parts[i].cell_id = get_next_cell();
    parts[i]._valid = FALSE;
  }  
  free(pos_x);
  free(pos_y);
  free(pos_z);
}

int main(int argc, char **argv){

  size_t padded_size = N_PARTS + 1000*TRADEQUEUE_SIZE;
  struct part* parts = (struct part*) malloc(sizeof(struct part) * padded_size);
  struct cell* cells = (struct cell*) malloc(sizeof(struct cell) * 1000);
  init_parts(parts, padded_size);
  for(int i = 0; i < 1000; i++){
    cells[i].id = i;
    init_cell(&cells[i], parts, padded_size);
  }
  free(parts);
  int counter = 0;
  for(int i = 0; i < cells[999].nparts; i++){
    if(cells[999].parts[i]._valid) counter++;
  }
  printf("parts in cell 999, nparts %i, counter %i\n", cells[999].nparts, counter);
  double start = omp_get_wtime();
#pragma omp parallel default(none) shared(cells)
{
  #pragma omp master
  {
    for(int i = 0; i < 1000; i++){
      #pragma omp task firstprivate(i) depend(inout: cells[i])
      {
          timestep_task(&cells[i]);
      }
    }

//    for(int i = 0; i < 1000; i++){
//      #pragma omp task firstprivate(i) depend(inout: cells[i])
//      {
//          self_task(&cells[i]);
//      }
//    }
  }
}
  double end = omp_get_wtime();

  printf("Omp version timestep and self in %fs\n", end-start);

  //Cleanup
  for(int i =0; i < 1000; i++){
    delete_cell(&cells[i]);
  }
  free(cells);
}
