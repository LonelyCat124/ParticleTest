#include <cstdio>
#include "legion.h"
#include <random>
#include  <cmath>
#include "read_file.h"

using namespace Legion;

enum TaskID{
  MAIN_TASK,
  INIT_TASK,
  TIMESTEP_TASK,
  SELF_TASK,
};

enum ParticleType{
  POS_X,
  POS_Y, 
  POS_Z,
  VEL_X,
  VEL_Y,
  VEL_Z,
  ACC_X,
  ACC_Y,
  ACC_Z,
  CELL,
  _VALID,
};

//We're going to use a cutoff of 0.1 and dimensions of 1x1x1 for easyness
#define CUTOFF 0.1
#define CUTOFF2 0.01
#define MASS 1
#define XDIM 1.0
#define YDIM 1.0
#define ZDIM 1.0
#define TIMESTEP 0.01

#define TRADEQUEUE_SIZE 100

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
  int64_t cell = get_cell_1d(x_cell, y_cell, z_cell);
  return cell;
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

Domain::DomainPointIterator copy_and_move_one(Domain::DomainPointIterator input_iterator){
  Domain::DomainPointIterator rval(input_iterator);
  rval++;
  return rval;
}

void self_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime) {
    assert(regions.size() == 1);
    assert(task->regions.size() == 1);
    FieldID fid[task->regions[0].privilege_fields.size()];
    int count = 0;
    for(std::set<FieldID>::iterator it = task->regions[0].privilege_fields.begin(); it != task->regions[0].privilege_fields.end(); it++){
        fid[count] = *it;
        count++;
    }
    const FieldAccessor<READ_WRITE, double, 1> pos_x(regions[0], fid[0]);
    const FieldAccessor<READ_WRITE, double, 1> pos_y(regions[0], fid[1]);
    const FieldAccessor<READ_WRITE, double, 1> pos_z(regions[0], fid[2]);
    const FieldAccessor<READ_WRITE, double, 1> vel_x(regions[0], fid[3]);
    const FieldAccessor<READ_WRITE, double, 1> vel_y(regions[0], fid[4]);
    const FieldAccessor<READ_WRITE, double, 1> vel_z(regions[0], fid[5]);
    const FieldAccessor<READ_WRITE, double, 1> acc_x(regions[0], fid[6]);
    const FieldAccessor<READ_WRITE, double, 1> acc_y(regions[0], fid[7]);
    const FieldAccessor<READ_WRITE, double, 1> acc_z(regions[0], fid[8]);
    const FieldAccessor<WRITE_DISCARD, int64_t, 1> cell(regions[0], fid[9]);
    const FieldAccessor<READ_WRITE, bool, 1> valid(regions[0], fid[10]);
    Domain dom = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
    for (Domain::DomainPointIterator pir(dom); pir; pir++){
      if(!valid[*pir]){
          continue;
      }
      double p1_x = pos_x[*pir];
      double p1_y = pos_y[*pir];
      double p1_z = pos_z[*pir];
      double fix = 0, fiy = 0, fiz = 0;
      for (Domain::DomainPointIterator pid = copy_and_move_one(pir); pid; pid++){
          if(!valid[*pid]){
              continue;
          }
          double p2_x = pos_x[*pid];
          double p2_y = pos_y[*pid];
          double p2_z = pos_z[*pid];

          double dx = p1_x - p2_x;
          double dy = p1_y - p2_y;
          double dz = p1_z - p2_z;

          double r2 = dx * dx + dy * dy + dz * dz;
          //If within cutoff
          if(r2 <= CUTOFF2){
            //Do some computation
//            printf("Found a neighbour\n");
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
          	acc_x[*pid] = acc_x[*pid] - fx;
          	acc_y[*pid] = acc_y[*pid] - fy;
          	acc_z[*pid] = acc_z[*pid] - fz;
          }
      }
      acc_x[*pir] = acc_x[*pir] + fix;
      acc_y[*pir] = acc_y[*pir] + fiy;
      acc_z[*pir] = acc_z[*pir] + fiz;
    }
}

void simple_timestepping_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime) {
    assert(regions.size() == 1);
    assert(task->regions.size() == 1);
//    printf("Number of accessible fields is %lu\n", task->regions[0].privilege_fields.size());
    FieldID fid[task->regions[0].privilege_fields.size()];
    int count = 0;
    for(std::set<FieldID>::iterator it = task->regions[0].privilege_fields.begin(); it != task->regions[0].privilege_fields.end(); it++){
        fid[count] = *it;
        count++;
    }
    const FieldAccessor<READ_WRITE, double, 1> pos_x(regions[0], fid[0]);
    const FieldAccessor<READ_WRITE, double, 1> pos_y(regions[0], fid[1]);
    const FieldAccessor<READ_WRITE, double, 1> pos_z(regions[0], fid[2]);
    const FieldAccessor<READ_WRITE, double, 1> vel_x(regions[0], fid[3]);
    const FieldAccessor<READ_WRITE, double, 1> vel_y(regions[0], fid[4]);
    const FieldAccessor<READ_WRITE, double, 1> vel_z(regions[0], fid[5]);
    const FieldAccessor<READ_WRITE, double, 1> acc_x(regions[0], fid[6]);
    const FieldAccessor<READ_WRITE, double, 1> acc_y(regions[0], fid[7]);
    const FieldAccessor<READ_WRITE, double, 1> acc_z(regions[0], fid[8]);
    const FieldAccessor<WRITE_DISCARD, int64_t, 1> cell(regions[0], fid[9]);
    const FieldAccessor<READ_WRITE, bool, 1> valid(regions[0], fid[10]);
    Domain dom = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
    for(Domain::DomainPointIterator pir(dom); pir; pir++){
        if(valid[*pir]){
            pos_x[*pir] = pos_x[*pir] + vel_x[*pir]*TIMESTEP;
            pos_y[*pir] = pos_y[*pir] + vel_y[*pir]*TIMESTEP;
            pos_z[*pir] = pos_z[*pir] + vel_z[*pir]*TIMESTEP;
            vel_x[*pir] = vel_x[*pir] + acc_x[*pir]*TIMESTEP;
            vel_y[*pir] = vel_y[*pir] + acc_y[*pir]*TIMESTEP;
            vel_z[*pir] = vel_z[*pir] + acc_z[*pir]*TIMESTEP;
        }
    }
}

void initialisation_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime) {
    assert(regions.size() == 1);
    assert(task->regions.size() == 1);
    printf("Number of accessible fields is %lu\n", task->regions[0].privilege_fields.size());
    FieldID fid[task->regions[0].privilege_fields.size()];
    int count = 0;
    for(std::set<FieldID>::iterator it = task->regions[0].privilege_fields.begin(); it != task->regions[0].privilege_fields.end(); it++){
        fid[count] = *it;
        count++;
    }
    const FieldAccessor<WRITE_DISCARD, double, 1> pos_x(regions[0], fid[0]);
    const FieldAccessor<WRITE_DISCARD, double, 1> pos_y(regions[0], fid[1]);
    const FieldAccessor<WRITE_DISCARD, double, 1> pos_z(regions[0], fid[2]);
    const FieldAccessor<WRITE_DISCARD, double, 1> vel_x(regions[0], fid[3]);
    const FieldAccessor<WRITE_DISCARD, double, 1> vel_y(regions[0], fid[4]);
    const FieldAccessor<WRITE_DISCARD, double, 1> vel_z(regions[0], fid[5]);
    const FieldAccessor<WRITE_DISCARD, double, 1> acc_x(regions[0], fid[6]);
    const FieldAccessor<WRITE_DISCARD, double, 1> acc_y(regions[0], fid[7]);
    const FieldAccessor<WRITE_DISCARD, double, 1> acc_z(regions[0], fid[8]);
    const FieldAccessor<WRITE_DISCARD, int64_t, 1> cell(regions[0], fid[9]);
    const FieldAccessor<WRITE_DISCARD, bool, 1> valid(regions[0], fid[10]);
    Rect<1> rect = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
    //Read positions from input
    double* x_positions = (double*) malloc(sizeof(double) * 100000);
    double* y_positions = (double*) malloc(sizeof(double) * 100000);
    double* z_positions = (double*) malloc(sizeof(double) * 100000);
    count =0;
    read_positions(x_positions, y_positions, z_positions);
    for(PointInRectIterator<1> pir(rect); pir(); pir++){
      if(count < 100000){
        double x_pos = x_positions[count];
        double y_pos = y_positions[count];
        double z_pos = z_positions[count];
        count++;
        pos_x[*pir] = x_pos;
        pos_y[*pir] = y_pos;
        pos_z[*pir] = z_pos;
        vel_x[*pir] = 0.0;
        vel_y[*pir] = 0.0;
        vel_z[*pir] = 0.0;
        acc_x[*pir] = 0.0;
        acc_y[*pir] = 0.0;
        acc_z[*pir] = 0.0;
        uint64_t cell_id = get_cell(x_pos, y_pos, z_pos);
        cell[*pir] = cell_id;
        valid[*pir] = true;
      }else{
        uint64_t cell_id = get_next_cell();
        cell[*pir] = cell_id;
        valid[*pir] = false;
        count++;
      }
    }
    free(x_positions);
    free(y_positions);
    free(z_positions);
}

void main_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime) {
    printf("Hello world!\n");
    int num_parts = 100000;
    
    //Initialise the particle array with extra space
    Rect<1> elem_rect(0, num_parts-1 + 1000*TRADEQUEUE_SIZE);
    IndexSpace particle_space = runtime->create_index_space(ctx, elem_rect);
    FieldSpace input_fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, input_fs);
        allocator.allocate_field(sizeof(double), POS_X);
        allocator.allocate_field(sizeof(double), POS_Y);
        allocator.allocate_field(sizeof(double), POS_Z);
        allocator.allocate_field(sizeof(double), VEL_X);
        allocator.allocate_field(sizeof(double), VEL_Y);
        allocator.allocate_field(sizeof(double), VEL_Z);
        allocator.allocate_field(sizeof(double), ACC_X);
        allocator.allocate_field(sizeof(double), ACC_Y);
        allocator.allocate_field(sizeof(double), ACC_Z);
        allocator.allocate_field(sizeof(int64_t), CELL);
        allocator.allocate_field(sizeof(bool), _VALID);
    }
    LogicalRegion particle_array = runtime->create_logical_region(ctx, particle_space, input_fs);

    //Setup the privileges and task for the initialisation task
    TaskLauncher init_launcher(INIT_TASK, TaskArgument(NULL, 0));
    RegionRequirement init_req(particle_array, WRITE_DISCARD, EXCLUSIVE, particle_array);
    init_req.add_field(POS_X);
    init_req.add_field(POS_Y);
    init_req.add_field(POS_Z);
    init_req.add_field(VEL_X);
    init_req.add_field(VEL_Y);
    init_req.add_field(VEL_Z);
    init_req.add_field(ACC_X);
    init_req.add_field(ACC_Y);
    init_req.add_field(ACC_Z);
    init_req.add_field(CELL);
    init_req.add_field(_VALID);
    init_launcher.add_region_requirement(init_req);
    //Run the initialisation task
    runtime->execute_task(ctx, init_launcher);

    //Set up the cell partition (for now just using an equal partition)
    Rect<1> cell_rect(0, 999);
    IndexSpace cell_space = runtime->create_index_space(ctx, cell_rect);
//    IndexPartition ip = runtime->create_equal_partition(ctx, particle_space, cell_space);
    IndexPartition ip = runtime->create_partition_by_field(ctx, particle_array, particle_array, CELL, cell_space);
    runtime->attach_name(ip, "ip");
    LogicalPartition cell_partition = runtime->get_logical_partition(ctx, particle_array, ip);
    runtime->attach_name(cell_partition, "cell_partition");



    ArgumentMap arg_map;
    IndexLauncher timestep_launcher(TIMESTEP_TASK, cell_space, TaskArgument(NULL,0), arg_map);
    RegionRequirement timestep_req(cell_partition, 0, READ_WRITE, ATOMIC, particle_array);
    timestep_req.add_field(POS_X);
    timestep_req.add_field(POS_Y);
    timestep_req.add_field(POS_Z);
    timestep_req.add_field(VEL_X);
    timestep_req.add_field(VEL_Y);
    timestep_req.add_field(VEL_Z);
    timestep_req.add_field(ACC_X);
    timestep_req.add_field(ACC_Y);
    timestep_req.add_field(ACC_Z);
    timestep_req.add_field(CELL);
    timestep_req.add_field(_VALID);
    timestep_launcher.add_region_requirement(timestep_req);

    IndexLauncher self_task_launcher(SELF_TASK, cell_space, TaskArgument(NULL,0), arg_map);
    self_task_launcher.add_region_requirement(timestep_req);


    runtime->execute_index_space(ctx, timestep_launcher);
    runtime->execute_index_space(ctx, self_task_launcher);
    runtime->issue_execution_fence(ctx);
    Future start = runtime->get_current_time_in_microseconds(ctx);
    runtime->issue_execution_fence(ctx);
    for(int i = 0; i < 10; i++){
    runtime->execute_index_space(ctx, timestep_launcher);
    runtime->execute_index_space(ctx, self_task_launcher);
    runtime->issue_execution_fence(ctx);
    }
    Future end = runtime->get_current_time_in_microseconds(ctx);
    printf("Runtime for timestep and self was %fs\n", (double)(end.get_result<long long>()-start.get_result<long long>()) / 1000000.0); 

    //Cleanup
    runtime->destroy_logical_region(ctx, particle_array);
    runtime->destroy_field_space(ctx, input_fs);
    runtime->destroy_index_space(ctx, particle_space);
    runtime->destroy_index_space(ctx, cell_space);
}

int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(MAIN_TASK);
    
    {
        TaskVariantRegistrar registrar(MAIN_TASK, "main task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<main_task>(registrar, "Main task");
    }
    {
        TaskVariantRegistrar registrar(INIT_TASK, "init task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<initialisation_task>(registrar, "Init task");
    }
    {
        TaskVariantRegistrar registrar(TIMESTEP_TASK, "timestep task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<simple_timestepping_task>(registrar, "Timestep task");
    }
    {
        TaskVariantRegistrar registrar(SELF_TASK, "self task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<self_task>(registrar, "Self task");
    }
    return Runtime::start(argc, argv);
}
