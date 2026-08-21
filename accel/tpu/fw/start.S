/* start.S — reset entry for the PicoRV32 in rtl/cpu_subsys.sv.
 *
 * The core comes out of reset at PROGADDR_RESET = 0 with sp already set from
 * its STACKADDR parameter; both are re-established here from the linker script
 * so link.ld stays the single source of truth. There is no .data copy: the
 * image is loaded straight into the firmware RAM it runs from, so LMA == VMA.
 */
    .section .text.init, "ax", @progbits
    .global _start
    .type   _start, @function

_start:
    .option push
    .option norelax
    la      gp, __global_pointer$
    .option pop

    la      sp, _stack_top

    /* .bss = 0 */
    la      t0, _sbss
    la      t1, _ebss
1:  bgeu    t0, t1, 2f
    sw      zero, 0(t0)
    addi    t0, t0, 4
    j       1b

2:  call    main

    /* Raise `done`. tpu_top reports it to the host exactly where the scalar
     * unit's HALT used to, and drops cpu_run — which puts the core back in
     * reset, so the spin below is only a parking spot. */
    li      t0, 0x80000070
    sw      zero, 0(t0)
3:  j       3b

    .size   _start, . - _start
