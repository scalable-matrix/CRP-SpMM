CRPSPMM_INSTALL_DIR = ..

DEFS    = 
INCS    = -I$(CRPSPMM_INSTALL_DIR)/include
CFLAGS  = $(INCS) -Wall -g -std=gnu11 -O3 -fPIC $(DEFS)
LDFLAGS = -g -O3 -fopenmp
LIBS    = $(CRPSPMM_INSTALL_DIR)/lib/libcrpspmm.a

GENCODE_SM60  = -gencode arch=compute_60,code=sm_60
GENCODE_SM70  = -gencode arch=compute_70,code=sm_70
GENCODE_FLAGS = $(GENCODE_SM60) $(GENCODE_SM70)

CUDA_PATH   ?= /usr/local/cuda-11.0
NVCC        = nvcc
NVCCFLAGS   = -O3 -g --compiler-options -fPIC $(GENCODE_FLAGS)

ifeq ($(shell $(CC) --version 2>&1 | grep -c "icc"), 1)
CFLAGS  += -fopenmp -xHost
endif

ifeq ($(shell $(CC) --version 2>&1 | grep -c "gcc"), 1)
CFLAGS  += -fopenmp -march=native -Wno-unused-result -Wno-unused-function
LIBS    += -lgfortran -lm
endif

ifeq ($(strip $(USE_MKL)), 1)
DEFS    += -DUSE_MKL
CFLAGS  += -mkl
LDFLAGS += -mkl
endif

ifeq ($(strip $(USE_OPENBLAS)), 1)
OPENBLAS_INSTALL_DIR = ../../OpenBLAS-git/install
DEFS    += -DUSE_OPENBLAS
INCS    += -I$(OPENBLAS_INSTALL_DIR)/include
LDFLAGS += -L$(OPENBLAS_INSTALL_DIR)/lib
LIBS    += -lopenblas
endif

ifeq ($(strip $(USE_CUDA)), 1)
OBJS    += $(CU_OBJS)
DEFS    += -DUSE_CUDA
LDFLAGS += -L$(CUDA_PATH)/lib64
LIBS    += -lcuda -lcudart -lcublas -lcusparse -lcusolver -lcurand
endif

C_SRCS 	= $(wildcard *.c)
C_OBJS  = $(C_SRCS:.c=.c.o)
EXES    = test_crpspmm.exe crpspmm_calc_partition.exe
# Delete the default old-fashion double-suffix rules
.SUFFIXES:

.SECONDARY: $(C_OBJS)

all: $(EXES)

%.c.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

test_crpspmm.exe: mmio.c.o mmio_utils.c.o test_crpspmm.c.o $(CRPSPMM_INSTALL_DIR)/lib/libcrpspmm.a
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

crpspmm_calc_partition.exe: mmio.c.o mmio_utils.c.o crpspmm_calc_partition.c.o 
	$(CC) $(LDFLAGS) -o $@ $^ 

clean:
	rm -f $(EXES) $(C_OBJS)