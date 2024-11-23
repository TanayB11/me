+++
title = "On \"The Case for the Reduced Instruction Set Computer\""
date = 2024-11-23
description = "🌳"
+++

> An assignment for CS254 (Advanced Computer Architecture) at UCSB

Personal computers were booming around the 1980s: Apple [released](https://en.wikipedia.org/wiki/Apple_II) the famous Apple II in 1977 and started [selling](https://en.wikipedia.org/wiki/Apple_Lisa) the Lisa in 1983. Meanwhile, IBM [released](https://en.wikipedia.org/wiki/IBM_Personal_Computer) the IBM Personal Computer in 1981. Around this time, complex instruction set computers (CISCs) dominated the market with VAX, x86, and related instruction sets.

Even today, CISCs [flaunt](https://en.wikipedia.org/wiki/X86_instruction_listings#Cryptographic_instructions) their huge instruction sets, which make it easy to perform additions, polynomial evaluations, dot products, and even cryptographic functions. They (usually) make programmers’ lives easier by providing a single instruction for common tasks: Function calls don't have to consist of explicit instructions to push arguments onto the stack, link a return address, and jump to the beginning of the function. In x86, all this functionality is wrapped up in an elegant `CALL`. Your code takes up less disk space, and life is good.

This argument is great for marketing teams at computer companies: They can just use “instruction set size” to sell computers. These companies have also been focused on related KPIs such as MIPS (millions of instructions per second) and MHz (GHz today), arguing that "more megahertz = more performance."

The problem is that **these metrics are a leaky proxy for performance**.

## Example: Matrix multiplication

For example, when multiplying a ton of matrices, many simple GPU cores that run at MHz speeds will be much faster than a few complex CPU cores running at GHz speeds. Although the GPU runs slower, it has thousands of cores that can each compute dot products in parallel. CPUs have fewer cores, so their throughput is much more limited.

<center>
    <img src="/images/2024-11-23-on-the-case-for-the-risc/cpu-vs-gpu.png" width="70%"
    style="border-radius: 0.5em;"/>
</center>
<center><em>(CPU: <a href="https://github.com/anycore/anycore-riscv-src">AnyCore</a>)</em></center>

## The Only Way: Evaluate your processor by running the program

More available instructions add complexity to the hardware, and adding unnecessary instructions can actually *impair* performance.

That's why application benchmarks are so important.

As consumers, we like to evaluate the metrics that companies print on the box. But it's dangerous to just pick a few metrics and blindly optimize for them. We should really be asking, “How fast/efficiently will this processor run my programs?”

As computer architects David Patterson and David Ditzel write in "The Case for the Reduced Instruction Set Computer” [^1]

> Unfortunately, the primary goal of a computer company is not to design the most cost-effective computer; the primary goal of a computer company is to make the most money by selling computers.”

## The Case for the Reduced Instruction Set Computer [^1]

Most CISCs have a host of problems: Here are the main ones, as outlined in Patterson and Ditzel's paper:

### Upward compatibility

CISCs add instructions over time, making the instruction set architecture (ISA) increasingly bloated. Once architects commit to adding an instruction to the ISA, they've effectively locked themselves in for life: Binaries will start using the new instructions; to ensure upward compatibility, the instruction won't disappear in future versions of the instruction set.

Consequently, it is *very* hard to get rid of badly designed instructions. Like a tower crumbling under its own weight, the architecture becomes increasingly difficult to support.

### Irrational implementations

One of the worst offenders in CISC design is the irrational implementation: An instruction that probably shouldn't exist. For some reason, such as marketing, an unnecessarily complex instruction finds its way into a CISC ISA. In the paper,

> One example was discovered by Peuto and Shustek for the IBM 370 [Peuto,Shustek77]; they found that a sequence of load instructions is faster than a load multiple instruction for fewer than 4 registers.

One dedicated instruction isn't necessarily faster than several smaller ones. And because architects often demand upward compatibility, these irrational implementations often stay in the ISA.

RISC simply doesn't have this problem: Although RISCs sometimes support extensions with special-purpose instructions, the base instruction set is simple and rarely changes. It's against the RISC philosophy to introduce anything that might *possibly* be an irrational implementation.

### Compilers don't use all the instructions

Of course, CISCs do have useful special-purpose instructions. However, compilers often neglect many of them. Thus, many CISC instructions are effectively useless. In the paper, Patterson and Ditzel write,

> Measurements of a particular IBM 360 compiler found that 10 instructions accounted for 80% of all instructions executed, 16 for 90%, 21 for 95%, and 30 for 99% [Alexander75].

If users (programmers and compiler writers) rarely use a CISC instruction, is it worth keeping it? After all, limiting the number of instructions in an instruction set has several advantages, such as:

### Design + fab time effects

One of the mantras of the tech industry is "move fast, break things. While it's probably a bad idea to break things in computer architecture (hardware has to be correct), the companies that win are often the ones that iterate the fastest.

Design iteration is necessarily slower for CISCs than for RISCs. Complex instructions introduce more complex logic into the CPU. Hardware becomes more difficult to design and test. Architects are more prone to design errors, and fab yields may even be lower due to the added complexity.

Ultimately, a CISC chip will cost more to produce than a similar RISC chip.

### Better use of chip area

According to Moore's Law, VLSI design has been getting better and better. So maybe it *does* make sense to opt for a CISC. However, as the paper argues,

> As VLSI technology improves, the RISC architecture can  always stay one step ahead of the comparable CISC. When the CISC becomes realizable on a single chip, the RISC will have the silicon area to use pipelining techniques; when the CISC gets pipelining  the RISC will have on chip caches, etc.

<center>
    <img src="/images/2024-11-23-on-the-case-for-the-risc/cisc-risc-scaling.png" width="70%"
    style="border-radius: 0.5em;"/>
</center>
<center><em> A theoretical, completely scientific diagram of performance vs. number of transistors.  </em></center>

Maybe there is a region on the curve where CISC outperforms RISC. But in the long run, RISC is likely to be the better choice. As we get more transistors, the paper argues, we're better off spending our silicon on improving other parts of the processor rather than adding increasingly complex instruction logic. Our branch predictors can improve with larger branch history tables, and we can even go multicore!

After all, if many CISC instructions don't even make sense to use, and if compilers only use a few instructions to begin with, it makes more sense to spend our silicon budget on things that are *guaranteed* to be useful.

## RISC is great (but not a panacea)

Of course, CISCs *can* be done well. If there is a special-purpose, complex instruction that allows for huge compiler optimizations, then that instruction is probably worth including. And RISC does have drawbacks: A simpler instruction set, for example, usually requires more memory accesses (which can slow performance) [^2].

In principle, however, RISC makes sense for most designs. Apple has been demonstrating this with their Apple Silicon: Transistor count scales roughly quadratically with CPU pipeline width. By choosing to build on ARM (RISC), Apple could afford to spend their silicon budget on increasing the pipeline width. And by building these slower-but-wider machines, Apple has leapfrogged its competition with some of the most powerful and efficient consumer machines.

As an engineer, I like to solve problems by looking at the simplest solutions first and adding complexity only when necessary. That's why I'm a RISC fan: The philosophy is inherently simpler.

---

[^1]: David A. Patterson and David R. Ditzel. 1980. The case for the reduced instruction set computer. SIGARCH Comput. Archit. News 8, 6 (October 1980), 25–33. https://doi.org/10.1145/641914.641917

[^2]: https://cs.stanford.edu/people/eroberts/courses/soco/projects/risc/risccisc/
