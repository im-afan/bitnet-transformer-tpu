# Communication Interface

Inter-TPU link that lets one device write into a neighbor's DRAM, enabling multi-device
inference (scale-out). See [README](README.md) §4.

## 1. Purpose

Partition a model across several TPUs and move activations between them without host
involvement. A device produces a tile of activations, then pushes it directly into the
DRAM of the neighbor that owns the next stage — a one-sided **remote write**, the
hardware analog of a message-passing send. This backs the `WriteNeighbor` ISA op.

## 2. Topology

2D mesh — each TPU has **4 neighbors** (N, E, S, W), addressed by direction. A device
does not need global addressing; it only names a *direction* and an address in that
neighbor's DRAM.

```
        (N)
         │
 (W) ── TPU ── (E)
         │
        (S)
```

Edge devices have fewer live links; unconnected directions are marked absent in a config
register and a `WriteNeighbor` to them faults. Larger meshes tile this pattern; routing
beyond immediate neighbors (multi-hop) is **out of scope for v1** — only nearest-neighbor
writes are supported.

## 3. Transaction: `WriteNeighbor(neighbor, my_addr, neighbor_addr)`

One-sided put. The initiator reads `len` bytes from its own DRAM at `my_addr` and streams
them to the target's DRAM at `neighbor_addr`. The target's DMA engine accepts the stream
and writes it; the target CPU/scalar unit is **not interrupted per-word** — it polls a
completion flag or waits on a doorbell.

```
initiator                        target
─────────                        ──────
read my_addr (DMA) ─┐
                    ├─► link (flit stream: header + payload)
                    │        header = {dir, neighbor_addr, len}
                    └──────────────────► RX FSM ─► DMA write neighbor_addr
                                                   set doorbell / completion flag
```

### Flit format (suggested)

| Field    | Bits | Meaning                          |
| -------- | ---- | -------------------------------- |
| `dst`    | 2    | direction the sender used        |
| `addr`   | 32   | target DRAM byte address         |
| `len`    | 16   | payload bytes                    |
| `seq`    | 8    | ordering / duplicate detection   |
| payload  | ×    | data words                       |

## 4. Link layer

- **Physical:** a parallel or serial link per edge (LVDS pairs / GTs, board-dependent —
  deferred to `constraints/`). Width is a `PARAM`.
- **Flow control:** credit-based. The receiver advertises free RX-buffer credits; the
  sender only injects flits it has credits for, so a slow receiver back-pressures rather
  than dropping data.
- **Ordering / integrity:** per-link sequence numbers + CRC per flit; a failed CRC forces
  retransmit of that flit. Writes from a single sender to a single neighbor are delivered
  in order.
- **Clock domain:** the link runs in its own clock domain; an async FIFO crosses into the
  device's compute clock at each end.

## 5. Interaction with the memory system

`WriteNeighbor` targets **DRAM**, not scratchpad (consistent with the ISA's Comms/Memory
group operating on RAM addresses). Typical scale-out step:

1. Device *A* finishes its layers, results sit in scratchpad.
2. `WriteMemory` spills the boundary activations to *A*'s DRAM.
3. `WriteNeighbor` pushes them into *B*'s DRAM.
4. *B* sees the doorbell, `ReadMemory` pulls them into *B*'s scratchpad, continues.

Because writes are one-sided, *B* needs a synchronization signal to know data has landed:
a per-neighbor **doorbell register** (incremented on completion) that *B*'s scalar unit
polls or blocks on.

## 6. Interface

| Signal          | Dir | Width | Meaning                                  |
| --------------- | --- | ----- | ---------------------------------------- |
| `nb_start`      | in  | 1     | begin a WriteNeighbor                    |
| `nb_dir`        | in  | 2     | target direction                         |
| `src_addr`      | in  | 32    | local DRAM source                        |
| `dst_addr`      | in  | 32    | neighbor DRAM destination                |
| `len`           | in  | 16    | bytes                                    |
| `nb_done`       | out | 1     | local send complete (credits returned)   |
| `doorbell[4]`   | out | —     | per-direction inbound-completion flags   |
| `link_absent`   | out | 4     | which directions are unconnected         |

## 7. Open questions

- Link PHY choice and width — gated on target board.
- Whether to add a lightweight multi-hop router for meshes larger than can be filled by
  nearest-neighbor partitioning.
- Reliability model: is per-flit CRC+retransmit enough, or is an end-to-end ack per
  `WriteNeighbor` needed for the partitioning scheme.
- Whether the doorbell should raise an interrupt to the scalar unit or remain poll-only.
