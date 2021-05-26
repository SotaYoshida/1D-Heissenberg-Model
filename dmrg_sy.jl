#!/usr/bin/env julia
using Arpack
using LinearAlgebra
using TimerOutputs

# Data structures to represent the block and enlarged block objects.
struct Block
    length::Int
    basis_size::Int
    op_dict::Dict{Symbol,AbstractMatrix{Float64}}
end

struct EnlargedBlock
    length::Int
    basis_size::Int
    op_dict::Dict{Symbol,AbstractMatrix{Float64}}
end

"""
    isvalid(block::Union{Block,EnlargedBlock})

The basis size must match the dimension of each operator matrix.
"""
function isvalid(block::Union{Block,EnlargedBlock}) 
    all(op -> size(op) == (block.basis_size, block.basis_size),
        values(block.op_dict))
end

"""
       H2(Sz1, Sp1, Sz2, Sp2)

Given the operators S^z and S^+ on two sites in two blocks (sys/env), 
returns a Kronecker product representing the corresponding term in the Hamiltonian.
"""
function H2(Sz1, Sp1, Sz2, Sp2;J=1.0,Jz=1.0)  # two-site part of H
    return (J/2.0) * (kron(Sp1, Sp2') + kron(Sp1', Sp2)) + Jz * kron(Sz1, Sz2)
end

function spbHkron!(spbH,tH,sys_enl_op,I_sys_enl,env_enl_op,I_env_enl,to;
                   J=1.0,Jz=1.0)
    #superblock_hamiltonian = kron(sys_enl_op[:H], I_env_enl)
    # + kron(I_sys_enl, env_enl_op[:H]) 
    # + H2(sys_enl_op[:conn_Sz], sys_enl_op[:conn_Sp],
    #      env_enl_op[:conn_Sz], env_enl_op[:conn_Sp])
    mykron!(spbH,sys_enl_op[:H], I_env_enl)
    mykron_add!(spbH,I_sys_enl,env_enl_op[:H])
    Sz1=sys_enl_op[:conn_Sz]; Sp1=sys_enl_op[:conn_Sp]
    Sz2=env_enl_op[:conn_Sz]; Sp2=env_enl_op[:conn_Sp]
    mykron_add!(spbH,Sp1,Sp2';fac=0.5*J)
    mykron_add!(spbH,Sp1',Sp2;fac=0.5*J)
    mykron_add!(spbH,Sz1, Sz2;fac=Jz)
    return nothing
end

function mykron!(ret,A,B)
    i1,j1 = size(A);i2,j2= size(B)
    @inbounds @simd for j= 1:j1
        for i = 1:i1
            Mat = @views ret[(i-1)*i2+1:i*i2,(j-1)*j2+1:j*j2]
            Mat .= A[i,j] .* B
        end
    end
    return nothing
end
function mykron_add!(ret,A,B;fac=1.0)
    i1,j1 = size(A);i2,j2= size(B)
    @inbounds @simd for j= 1:j1
        for i = 1:i1
            coeff = fac .* A[i,j]
            Mat = @views ret[(i-1)*i2+1:i*i2,(j-1)*j2+1:j*j2]
            Mat .+= coeff .* B
        end
    end
    return nothing
end


"""
    enlarge_block(block::Block)

Enlarges the given Block by a single site => EnlargedBlock
"""
function enlarge_block(block::Block)
    mblock = block.basis_size
    o = block.op_dict
    I1 = Matrix{Float64}(I,model_d,model_d)
    I_block = Matrix{Float64}(I,mblock,mblock)
    enlarged_op_dict = Dict{Symbol,AbstractMatrix{Float64}}(
        :H => kron(o[:H], I1) + kron(I_block, H1) + H2(o[:conn_Sz], o[:conn_Sp], Sz1, Sp1),
        :conn_Sz => kron(I_block, Sz1),:conn_Sp => kron(I_block, Sp1), )
    return EnlargedBlock(block.length + 1,
                         block.basis_size * model_d,
                         enlarged_op_dict)
end

"""
    rotate_and_truncate(operator, U)

Transforms the operator to the new (possibly truncated) basis given by `U`.
U^TOU
"""
function rotate_and_truncate(operator, U)
    return U' * (operator * U)
end

function make_Hermite!(spbH,Dim)
    @inbounds @simd for i = 1:Dim
        for j=i+1:Dim
            tmp = 0.5 * (spbH[i,j] + spbH[j,i])
            spbH[i,j] = tmp; spbH[j,i]=tmp
        end
    end
    return nothing
end

"""
    single_dmrg_step(sys::Block, env::Block, m::Int)

Performs a single DMRG step using `sys` as the system and `env` as the
environment, keeping a maximum of `m` states in the new basis.

spbH = (spbH +spbH')/2 is needed to ensure Hermitian
"""
function single_dmrg_step(sys::Block, env::Block, m::Int, spbH, tmpH, to)    
    @assert isvalid(sys); @assert isvalid(env)
    @timeit to "enlarge" begin
        # Enlarge each block by a single site.
        sys_enl = enlarge_block(sys)
        if sys === env
            env_enl = sys_enl
        else
            env_enl = enlarge_block(env)
        end
    end
    @assert isvalid(sys_enl); @assert isvalid(env_enl)

    @timeit to "superblock" begin
        # Construct the superblock Hamiltonian.
        m_sys_enl = sys_enl.basis_size
        m_env_enl = env_enl.basis_size
        sys_enl_op = sys_enl.op_dict
        env_enl_op = env_enl.op_dict
        I_sys_enl = Matrix{Float64}(I,m_sys_enl,m_sys_enl)
        I_env_enl = Matrix{Float64}(I,m_env_enl,m_env_enl)
        tDim = m_sys_enl * m_env_enl
        subH = @views spbH[1:tDim,1:tDim]
        tH   = @views tmpH[1:tDim,1:tDim]
        spbHkron!(subH,tH,sys_enl_op,I_sys_enl,
                  env_enl_op,I_env_enl,to)        
        make_Hermite!(subH,m_sys_enl^2)
    end
    @timeit to "Schur decomposition" begin
        @timeit to "eigs" (energy,), psi0 = eigs(subH, nev=1, which=:SR)
        # rho: reduced density matrix of the system (by tracing out the "env")
        psi0 = transpose(reshape(psi0, (env_enl.basis_size, sys_enl.basis_size)))
        tDim = min(tDim,size(psi0)[1])
        rho = @views tmpH[1:tDim,1:tDim]
        mul!(rho,psi0,psi0')
        make_Hermite!(rho,tDim)
        fact = eigen(rho)
        evals, evecs = fact.values, fact.vectors
        permutation = sortperm(evals, rev=true)
    
        my_m = min(length(evals), m)
        indices = @views permutation[1:my_m]
        U = @views evecs[:,indices]
        
        truncation_error = 1.0 - sum(evals[indices])
        
        # Rotate and truncate each operator.
        new_op_dict = Dict{Symbol,AbstractMatrix{Float64}}()
        for (name, op) in sys_enl.op_dict
        new_op_dict[name] = rotate_and_truncate(op, U)
        end    
        newblock = Block(sys_enl.length, my_m, new_op_dict)
    end

    return newblock, energy, truncation_error
end

function infinite_system_algorithm(L::Int, m::Int,initial_block,to)
    block = initial_block
    spbH = zeros(Float64,(2*m)^2,(2*m)^2) # superblock Hamiltonian
    tmpH = zeros(Float64,(2*m)^2,(2*m)^2) # matrix for efficient kron!()
    while 2 * block.length < L
        block, energy, trerr = single_dmrg_step(block, block, m, spbH, tmpH, to)
        println("L = ", block.length * 2 + 2, "\t",
                "truncation err ", trerr,"\t",
                "E/L = ", energy / (block.length * 2))
    end
end

function finite_system_algorithm(L::Int, m_warmup::Int,
                                 initial_block, m_sweep_list,to
                                 ;verbose=false)
    if m_sweep_list==[]; m_sweep_list=[m_warmup];end
    @assert iseven(L)
    @timeit to "warmup" begin
        spbH = zeros(Float64,(2*m_warmup)^2,(2*m_warmup)^2) # superblock Hamiltonian
        tmpH = zeros(Float64,(2*m_warmup)^2,(2*m_warmup)^2) # matrix for efficien kron!()
    
        # "disk" storage for Block objects
        block_disk = Dict{Tuple{Symbol,Int},Block}()    
        # Use the infinite system algorithm to build up
        # to desired size. Each time we construct a block,
        # we save it for future reference as both a left
        # (:l) and right (:r) block, as the infinite system algorithm
        # assumes the environment is a mirror image of the system.
        block = initial_block
        block_disk[:l, block.length] = block
        block_disk[:r, block.length] = block
        while 2 * block.length < L
            #println(graphic(block, block))
            block, energy,terr = single_dmrg_step(block, block, m_warmup, spbH, tmpH, to)
            tL = block.length * 2
            if verbose; println("warmup:$tL E/L = ", energy / tL);end
            block_disk[:l, block.length] = block
            block_disk[:r, block.length] = block
        end
    end
    @timeit to "sweep" begin
        sys_label, env_label = :l, :r
        sys_block = block # rename 
        block = Block(0, 0, Dict{Symbol,AbstractMatrix{Float64}}())        
        for m in m_sweep_list
            while true
                # Load the appropriate environment block from "disk"
                env_block = block_disk[env_label, L - sys_block.length - 2]
                if env_block.length == 1 # "reverse" at the end
                    sys_block, env_block = env_block, sys_block
                    sys_label, env_label = env_label, sys_label
                end
                # Perform a single DMRG step.
                #println(graphic(sys_block, env_block, sys_label))
                sys_block, energy, trerr = single_dmrg_step(sys_block, env_block, m, spbH, tmpH, to)
                if verbose;println("E/L = ", energy / L);end
                # Save the block from this step to disk.
                block_disk[sys_label, sys_block.length] = sys_block
                # Check whether we just completed a full sweep.
                if sys_label == :l && 2 * sys_block.length == L
                    println("L $L E/L = ", energy / L)
                    break  # escape from the "while true" loop
                end
            end
        end
    end
end

# Heisenberg chain
const model_d = 2  # single-site basis size
const Sz1 = [0.5 0.0; 0.0 -0.5]  # single-site S^z
const Sp1 = [0.0 1.0; 0.0 0.0]  # single-site S^+
const H1 = [0.0 0.0; 0.0 0.0]  # single-site portion of H is zero
function main(;L=100,m=20,do_sweep=false,m_sweep_list=[])
    to = TimerOutput()
    initial_block = Block(1, model_d, Dict{Symbol,AbstractMatrix{Float64}}(
        :H => H1,
        :conn_Sz => Sz1,
        :conn_Sp => Sp1,
    ))
    if do_sweep==false
        infinite_system_algorithm(L,m,initial_block,to)
    else
        finite_system_algorithm(L,m,initial_block,m_sweep_list,to)
    end
    #show(to, allocations = true,compact = false);println("") 
end


const sL = 100
const sm = 20

main(;L=sL,m=sm)
main(;L=sL,m=sm,do_sweep=true)
for sL = 10:2:200
    main(;L=sL,m=sm,do_sweep=true)
end


