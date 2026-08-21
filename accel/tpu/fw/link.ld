/* link.ld — the firmware RAM in rtl/cpu_subsys.sv: 2**FW_AW words at 0.
 *
 * Code, data and stack share the one 16 KB block, which is also the whole image
 * the host loads over 'I'. Keep _stack_top equal to cpu_subsys.sv's STACKADDR
 * (0x0000_0000 + (1 << (FW_AW + 2))) and LENGTH equal to its size.
 */
OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 16K
}

SECTIONS
{
    .text : {
        KEEP(*(.text.init))     /* _start must land at PROGADDR_RESET = 0 */
        *(.text .text.*)
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
        . = ALIGN(4);
    } > RAM

    .data : {
        . = ALIGN(4);
        __global_pointer$ = . + 0x800;
        *(.data .data.*)
        *(.sdata .sdata.*)
        . = ALIGN(4);
    } > RAM

    .bss (NOLOAD) : {
        . = ALIGN(4);
        _sbss = .;
        *(.sbss .sbss.*)
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    _stack_top = ORIGIN(RAM) + LENGTH(RAM);

    /DISCARD/ : { *(.comment) *(.eh_frame) *(.riscv.attributes) }
}
