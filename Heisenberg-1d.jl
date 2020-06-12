# Heisenberg-1d.jl
# The MIT License (MIT)
# Copyright (c) 2020 Sota Yoshida

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

using LinearAlgebra
using ArgParse
using SIMD
#using Arpack
#using SparseArrays
#using KrylovKit
#using BenchmarkTools
#using Profile

function prodS(vals::Array{Int64,1},numS::Int64,N::Int64)
    tlist = copy(vals)
    for i = 1:N-1
        nlist = []
        for tmp in tlist
            for val in vals
                if typeof(tmp) == Int
                    ctmp = [tmp]
                else
                    ctmp = copy(tmp)
                end
                push!(nlist,append!(ctmp,[val]))
            end
        end
        tlist = nlist
    end
    Dim = numS^N
    if length(tlist) != Dim
        println("These must be the same: length(tlist)=",length(tlist), " NumS^N=", Dim)
    end
    return tlist
end

function makeV(S::Float64,N::Int64)
    vals=collect(-Int64(2*S):2:Int64(2*S))
    numS = length(vals)
    prodS(vals,numS,N)
end


function convcheck(k::Int64,numstate::Int64,tDim::Int64,alphas::FA,betas::FA,en::FA,tol=1.e-14) where{FA<:Array{Float64,1}}
    if k < numstate
        return en,false
    else
        T = zeros(Float64,k,k)
        for i = 1:k
            T[i,i] =alphas[i]
        end
        for j = 2:k
            i = j-1
            T[i,j] = betas[i];T[j,i] = betas[i]
        end
        iend = minimum([tDim,numstate])

        nen = eigvals(T)[1:iend]
        if k == numstate; return nen,false; end      
        dif = maximum( abs.(en[1:iend] .- nen[1:iend]))
        if dif < tol
            return nen,true
        else
            return nen,false
        end
    end
end

function operateH(v::Array{Float64,1},t::Float64,U::Float64,
                  non0s::Array{Array{Int64,1},1},facs::Array{Array{Float64,1},1}
                  ,Ufacs::Array{Float64,1},tDim::Int64)
    ### diagonal part
    w = Ufacs .* v
    ### non-diagonal part
    tvec = 0.0 * v
    for ith = 1:tDim-1
        for (ind, jth) in enumerate(non0s[ith])
            tvec[jth] += 0.5* t * facs[ith][ind] * v[ith]
            tvec[ith] += 0.5* t * facs[ith][ind] * v[jth]
        end
    end
    w += tvec
    return w
end

function Lanczos(t::Float64,U::Float64,non0s::AAI,facs::AAF,Ufacs::FA,tDim::Int64,
                 numstate::Int64,maxiter=1000) where{FA<:Array{Float64,1},
                                                     AAI<:Array{Array{Int64,1},1},
                                                     AAF<:Array{Array{Float64,1},1}}
    vkm1 = zeros(Float64,tDim)
    vk   = zeros(Float64,tDim); vk[1]=1.0
    betakm1 = 0.0; betak=0.0; alphas = Float64[]; betas = Float64[]
    en = [1.0]; TF = false

    for k = 1:maxiter        
        w = operateH(vk,t,U,non0s,facs,Ufacs,tDim)  - betak .* vkm1
        alphak = dot(vk,w) 
        push!(alphas,alphak)
        en,TF = convcheck(k,numstate,tDim,alphas,betas,en)
        if TF;println(" En=",en[1:minimum([numstate,tDim])]," Lanczos it=$k"); break; end
        if k==maxiter
            try;println("En=",en[1:minimum([tDim,numstate])]);catch;println("En=$en");end
            #println("vec=$vk")
        end
        w = w - alphak .* vk 
        betakm1 = betak; betak = sqrt(dot(w,w)); push!(betas,betak)
        if betak == 0.0
            break
        end
        vkm1 = vk; vk = w/betak
    end
end


function calcfac(S::F,Sz_cre::F,Sz_ani::F) where{F<:Float64}
    sqrt( (S*(S+1.0) - Sz_cre*(Sz_cre+1.0)) * (S*(S+1.0) - Sz_ani*(Sz_ani-1.0)))
end
   
function neighbor(ret::IA,dif::IA,N::Int64) where{IA<:Array{Int64,1}} 
    ij = Int64[ ]
    for i=1:length(dif)
        if dif[i] != 0
            push!(ij,i)
        end
    end
    if abs(ij[1]-ij[2])== 1
        if dif[ij[1]] == 1
            ret[1]= 1; ret[2]=ij[1]; ret[3]=ij[2]
        else
            ret[1]= 1; ret[3]=ij[1]; ret[2]=ij[2]
        end
    else
        ret[1]=0 ; ret[2]=0; ret[3]=0
    end
    nothing
end

function neighborPBC(ret::IA,dif::IA,N::Int64) where{IA<:Array{Int64,1}} 
    ij = Int64[ ]
    for i=1:length(dif)
        if dif[i] != 0
            push!(ij,i)
        end
    end
    if abs(ij[1]-ij[2])== 1 || abs(ij[1]-ij[2])==N-1 
        if dif[ij[1]] == 1
            ret[1]= 1; ret[2]=ij[1]; ret[3]=ij[2]
        else
            ret[1]= 1; ret[3]=ij[1]; ret[2]=ij[2]
        end
    else
        ret[1]=0 ; ret[2]=0; ret[3]=0
    end
    nothing
end

function connectable(tf::IT,dif::IT) where{IT<:Array{Int64,1}}
    tf[1] = ifelse( sum( abs.(dif)) != 4 || sum(dif) != 0, 0,1)
    nothing 
end

function main(t::F,U::F,S::F,N::I,numstate::I,PBC::I) where{F<:Float64,I<:Int64}
    basisvec = makeV(S,N); println("Full Dim.=",length(basisvec))
    
    Slist = [ i for i=-S*N:1.0:S*N]
    vecsSz = [ [] for i=1:length(Slist)]
    for tvec in basisvec
        push!(vecsSz[Int64(0.5 * sum(tvec)+S*N)+1],tvec)
    end

    for i = 1:length(Slist)
        Sz = Slist[i]
        #if Sz != 0.0; continue ;end            
        tvec = vecsSz[i]        
        tDim = length(tvec)

        non0s=[Int64[] for ith=1:tDim]
        facs=[Float64[] for ith=1:tDim]
        #println("typeof(non0s) ",typeof(non0s), " typeof(facs) ", typeof(facs))
        Ufacs=zeros(Float64,tDim)
        tfac = 0.0; ret=zeros(Int64,3)
        print("Sz = $Sz tDim=$tDim\t")
        Fneighbor = ifelse(PBC==1, neighborPBC, neighbor)
        r = 0.0
        @inbounds for ith=1:tDim
            for j = 1:N-1
                Ufacs[ith] += 0.25*(tvec[ith][j]*tvec[ith][j+1])
            end
            Ufacs[ith] += ifelse(PBC==1, 0.25*tvec[ith][N]*tvec[ith][1],0)
        end
        @inbounds for jth=2:tDim
            @simd for ith = 1:jth
                connectable(ret,tvec[ith]-tvec[jth])                
                if ret[1] == 1
                    Fneighbor(ret,tvec[ith]-tvec[jth],N)
                    if ret[1]==1
                        push!(non0s[ith],jth)
                        push!(facs[ith],calcfac(S,0.5*tvec[jth][ret[2]],0.5*tvec[jth][ret[3]]))
                        ###  #Sz_cre = 0.5 * ket[cre]; Sz_ani = 0.5 * ket[ani]
                    end ## cre = ret[2]; ani=ret[3]
                end
            end       
        end
        if tDim == 1; println("En=$Ufacs");end
        Lanczos(t,U,non0s,facs,Ufacs,tDim,numstate)
    end
    nothing
end

function run()
    for (arg,val) in parsed_args
        print("$arg  => $val \t")
    end
    print("\n")
    t = parsed_args["t"]; S = parsed_args["S"];N = parsed_args["N"]
    nums = parsed_args["nums"]; PBC = parsed_args["PBC"]
    U = t  # (for debug) This must be identical in the 1D Heisenberg Hamiltonian"
    @time main(t,U,S,N,nums,PBC)
end


s = ArgParseSettings()
@add_arg_table! s begin
    "--t"
        help = "coefficient in the 1D Heisenberg Hamiltonian"
        arg_type = Float64
        default = 1.0
    "--S"
        help = "Spin (in FLoat) 0.5(S=1/2), 1.0 (S=1), etc."
        arg_type = Float64
        default = 0.5
    "--N"
        help = "particle number N"
        arg_type = Int64
        default = 10
    "--nums"
        help = "How many states do you need?"
        arg_type = Int64
        default = 1
    "--PBC"
        help = "Periodic Boundary Condition 1:true 0:false"
        arg_type = Int64
        default = 1 
end

parsed_args = parse_args(s)
run()
