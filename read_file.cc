#include "read_file.h"
#include <hdf5.h>

#define FILENAME "test.hdf5"

void write_position(double* x_pos, double* y_pos, double* z_pos, int count){
  hid_t file_id = H5Fcreate(FILENAME, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t shape[1];
  shape[0] = count;
  hid_t space = H5Screate_simple(1, shape, NULL);
  hid_t dataset = H5Dcreate2( file_id, "x_pos" , H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t status = H5Dwrite(dataset ,H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x_pos);
  H5Dclose(dataset);
  hid_t dataset2 = H5Dcreate2( file_id, "y_pos" , H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset2 ,H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, y_pos);
  H5Dclose(dataset2);
  hid_t dataset3 = H5Dcreate2( file_id, "z_pos" , H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset3 ,H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, z_pos);
  H5Dclose(dataset3);
  H5Fclose(file_id);
}

void read_positions(double* x_pos, double* y_pos, double* z_pos){
   hid_t file_id = H5Fopen(FILENAME, H5F_ACC_RDONLY, H5P_DEFAULT);

   hid_t x_pos_t = H5Dopen2(file_id, "x_pos", H5P_DEFAULT);
   hid_t space = H5Dget_space(x_pos_t);
   hid_t ndims = H5Sget_simple_extent_ndims(space);


   hsize_t dims[1];
   H5Sget_simple_extent_dims(space, dims, NULL);
   int particle_count = dims[0];

   hsize_t shape[2];
   hsize_t offset[2];
   shape[0] = particle_count;
   shape[1] = 1;
   offset[0] = 0;
   offset[1] = 0;
   hsize_t rank = 1;

   hid_t memspace = H5Screate_simple(rank, shape, NULL);
   hid_t filespace = H5Dget_space(x_pos_t);
   H5Sselect_hyperslab( filespace, H5S_SELECT_SET, offset, NULL, shape, NULL);
   H5Dread(x_pos_t, H5T_NATIVE_DOUBLE,  memspace, filespace, H5P_DEFAULT, x_pos);
   H5Dclose(x_pos_t);

   hid_t y_pos_t = H5Dopen2(file_id, "y_pos", H5P_DEFAULT);
   memspace = H5Screate_simple(rank, shape, NULL);
   filespace = H5Dget_space(y_pos_t);
   H5Sselect_hyperslab( filespace, H5S_SELECT_SET, offset, NULL, shape, NULL);
   H5Dread(y_pos_t, H5T_NATIVE_DOUBLE,  memspace, filespace, H5P_DEFAULT, y_pos);
   H5Dclose(y_pos_t);

   hid_t z_pos_t = H5Dopen2(file_id, "z_pos", H5P_DEFAULT);
   memspace = H5Screate_simple(rank, shape, NULL);
   filespace = H5Dget_space(z_pos_t);
   H5Sselect_hyperslab( filespace, H5S_SELECT_SET, offset, NULL, shape, NULL);
   H5Dread(z_pos_t, H5T_NATIVE_DOUBLE,  memspace, filespace, H5P_DEFAULT, z_pos);
   H5Dclose(z_pos_t);

   H5Fclose(file_id);
}
