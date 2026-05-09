NVCC = nvcc -g -G
NVCC_STD = -std=c++11

LEGACY_GPUS = V100 T4
FORMAT_GPUS = A100 H100 4090 5090
GPU_TARGETS = $(LEGACY_GPUS) $(FORMAT_GPUS)

GPU_SM_V100 = sm_70
GPU_SM_T4 = sm_75
GPU_SM_A100 = sm_80
GPU_SM_H100 = sm_90
GPU_SM_4090 = sm_89
GPU_SM_5090 = sm_120

FORMAT_TARGETS = binary16 bf16 binary64 tf32 binary16-details

SRC_V100 = tc_test_numerics-V100.cu
SRC_BINARY16 = tc_test_numerics-T4-A100-binary16.cu
SRC_BINARY16_DETAILS = tc_test_numerics-T4-A100-binary16-details.cu
SRC_BF16 = tc_test_numerics-A100-bf16.cu
SRC_BINARY64 = tc_test_numerics-A100-binary64.cu
SRC_TF32 = tc_test_numerics-A100-tf32.cu

SUPPORTED_SMS = $(shell nvcc --list-gpu-code 2>/dev/null)
ALL_TARGETS =

define ADD_DEFAULT_TARGET
ifeq ($(strip $(SUPPORTED_SMS)),)
ALL_TARGETS += test-$(1)
else ifneq ($(filter $(GPU_SM_$(1)),$(SUPPORTED_SMS)),)
ALL_TARGETS += test-$(1)
endif
endef

$(foreach gpu,$(GPU_TARGETS),$(eval $(call ADD_DEFAULT_TARGET,$(gpu))))

all: $(ALL_TARGETS)

test-V100: $(SRC_V100)
	$(NVCC) -o $@ -arch=$(GPU_SM_V100) $(NVCC_STD) $<

test-T4: $(SRC_BINARY16)
	$(NVCC) -o $@ -arch=$(GPU_SM_T4) $(NVCC_STD) $<

define ADD_FORMAT_GPU_TARGET
test-$(1): $(addprefix test-$(1)-,$(FORMAT_TARGETS))
endef

$(foreach gpu,$(FORMAT_GPUS),$(eval $(call ADD_FORMAT_GPU_TARGET,$(gpu))))

test-%-binary16: $(SRC_BINARY16)
	$(NVCC) -o $@ -arch=$(GPU_SM_$*) $(NVCC_STD) $<

test-%-bf16: $(SRC_BF16)
	$(NVCC) -o $@ -arch=$(GPU_SM_$*) $(NVCC_STD) $<

test-%-binary64: $(SRC_BINARY64)
	$(NVCC) -o $@ -arch=$(GPU_SM_$*) $(NVCC_STD) $<

test-%-tf32: $(SRC_TF32)
	$(NVCC) -o $@ -arch=$(GPU_SM_$*) $(NVCC_STD) $<

test-%-binary16-details: $(SRC_BINARY16_DETAILS)
	$(NVCC) -o $@ -arch=$(GPU_SM_$*) $(NVCC_STD) $<

clean: $(addprefix clean-,$(GPU_TARGETS)) clean-result

define ADD_LEGACY_CLEAN
clean-$(1):
	rm -f test-$(1)
endef

$(foreach gpu,$(LEGACY_GPUS),$(eval $(call ADD_LEGACY_CLEAN,$(gpu))))

define ADD_FORMAT_GPU_CLEAN
clean-$(1):
	rm -f test-$(1)-*
endef

$(foreach gpu,$(FORMAT_GPUS),$(eval $(call ADD_FORMAT_GPU_CLEAN,$(gpu))))

clean-result:
	rm -f result-*
