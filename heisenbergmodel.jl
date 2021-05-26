using LinearAlgebra
#using ArgParse
using Base.Threads
using SIMD
using TimerOutputs
#using Arpack
#using SparseArrays

""" 
    main(tJ,S,L,nums,PBC)

-tJ:      coefficient in the Hamiltonian
-S:       spin
-L:       length of chain
-nums:    # of states of interest 
PBC:      oeriodic boundary condition
"""
function main(tJ,S,L,nums,PBC;
              Slist=[0],is_show=false,retbasis=false)
    to = TimerOutput()
    ret = [0,0]
    ### prep. for calc.
    @timeit to "prep." begin
        basisvec = makeV(S,L)
        print("L=$L Full Dim.=",length(basisvec),"\t")
        vec = basisvec[1]
        vecsSz = [ [copy(vec)]]; deleteat!(vecsSz[1],1)
        bits = Int64[]
        for tvec in basisvec
            bitarr_to_int!(tvec,ret)
            bit = ret[1]
            lup = count_ones(bit)
            ldown = length(tvec) - lup
            tSz = Int64(0.5*(lup-ldown))
            for (i,Sz) in enumerate(Slist)
                if tSz != Sz; continue;end
                push!(vecsSz[1],tvec)
                push!(bits,bit)
            end
        end
        basisvec = nothing 
    end
    
    i = 1
    tvec = vecsSz[1]; tDim = length(tvec)
    println("Dim.(Sz=0) $tDim")    
    solve_1dHeisenberg(tJ,S,L,nums,PBC,tDim,tvec,to)    
    if is_show;show(to, allocations = true,compact =false);println("");end

    if retbasis; return vecsSz[1];end
    return nothing
end

function bitarr_to_int!(arr::Array{Bool,1},ret)
    ret[1] = 0
    ret[2] = 2^(length(arr)-1)
    for i in eachindex(arr)
        ret[1] += ret[2]*arr[i]
        ret[2] >>= 1
    end
    return nothing
end


function solve_1dHeisenberg(tJ,S,L,nums,PBC,tDim,tvec,to)
    ## potential term
    @timeit to "potential term" begin
        Ufacs=zeros(Float64,tDim)
        @inbounds for ith=1:tDim
            @simd for j = 1:L-1
                v1 = ifelse(tvec[ith][j]  ,1.0,-1.0)
                v2 = ifelse(tvec[ith][j+1],1.0,-1.0)                
                Ufacs[ith] += 0.25 * (v1 .*v2)
            end
            if PBC
                v1 = ifelse(tvec[ith][L],1.0,-1.0)
                v2 = ifelse(tvec[ith][1],1.0,-1.0)
                Ufacs[ith] += 0.25* (v1 .* v2)
            end
        end
    end
    ## hopping term
    @timeit to "hopping term" begin
        non0s=[ Int64[] for ith=1:tDim]
        facs=[Float64[] for ith=1:tDim]
        @inbounds @threads for ith=1:tDim
            TF = [false];ret=[0,0]
            tv1 = tvec[ith]
            t_non0s = non0s[ith]
            t_facs  = facs[ith]
            @inbounds for jth=ith+1:tDim                
                tv2=tvec[jth]
                connectable(TF,tv1,tv2)
                if TF[1]==false;continue;end
                neighbor(TF,ret,tv1,tv2,L,PBC)
                if TF[1]
                    v1  = 0.5 .* ifelse(tv2[ret[1]],1.0,-1.0)
                    v2  = 0.5 .* ifelse(tv2[ret[2]],1.0,-1.0)
                    fac = calcfac(S,v1,v2)
                    push!(t_non0s,jth)
                    push!(t_facs,fac)
                end
            end       
        end
    end
    @timeit to "Lanczos" Lanczos(tJ,L,non0s,facs,Ufacs,tDim,nums)
    return nothing
end

"""
    makeV(S,L)

preparing possible spin configurations
"""
function makeV(S,L)
    #sarr = collect(-Int64(2*S):2:Int64(2*S))
    sarr = [false,true]
    t = prodS(sarr,L)
end

"""
    prodS(sarr,N)

generating product of spin states to make configs.
"""
function prodS(sarr::Array{Bool,1},N::Int64)
    ln = length(sarr)
    tbasisvecs = copy(sarr)
    for i = 1:N-1
        nlist = []
        for tmp in tbasisvecs
            for val in sarr
                if typeof(tmp) == Bool
                    ctmp = [tmp]
                else
                    ctmp = copy(tmp)
                end
                push!(nlist,append!(ctmp,[val]))
            end
        end
        tbasisvecs = nlist
    end
    Dim = ln^N
    if length(tbasisvecs) != Dim;println("err prodS");exit();end
    return tbasisvecs
end


function calcfac(S::F,Sz_cre::F,Sz_ani::F) where{F<:Float64}
    sqrt( (S*(S+1.0) - Sz_cre*(Sz_cre+1.0)) * (S*(S+1.0) - Sz_ani*(Sz_ani-1.0)))
end
   
function neighbor(TF,ret,ta1,ta2,N::Int64,PBC=false) where{IA<:Array{Int64,1}} 
    ij = Int64[ ]    
    for i=1:length(ta1)
        dif = ta1[i]-ta2[i]
        if dif != 0
            push!(ij,i)
        end
    end
    if PBC ==false
        if abs(ij[1]-ij[2])== 1
            if ta1[ij[1]]-ta2[ij[1]] == 1
                TF[1]= true; ret[1]=ij[1]; ret[2]=ij[2]
            else
                TF[1]= true; ret[2]=ij[1]; ret[1]=ij[2]
            end
        else
            TF[1]=false ; ret[1]=0; ret[2]=0
        end
    else ### not checked
        if abs(ij[1]-ij[2])== 1 || abs(ij[1]-ij[2])==N-1 
            if dif[ij[1]] == 1
                TF[1]= true; ret[1]=ij[1]; ret[2]=ij[2]
            else
                TF[1]= true; ret[2]=ij[1]; ret[1]=ij[2]
            end
        else
            TF[1]=false ; ret[1]=0; ret[2]=0
        end
    end
    nothing
end

function connectable(tf,arr1,arr2) 
    s1 = 0; s2 = 0
    for (i,t1) in enumerate(arr1)
       t2 = arr2[i]
       dif = Int(t1)-Int(t2)
       s1 += abs(dif)
       s2 += dif
    end 
    tf[1] = ifelse( s1 != 2 || s2 != 0, false,true)
    return nothing 
end

function operateH(v::Array{Float64,1},t::Float64,
                  non0s::Array{Array{Int64,1},1},
                  facs::Array{Array{Float64,1},1},
                  Ufacs::Array{Float64,1},tDim::Int64)
    ### diagonal part
    w = Ufacs .* v
    ### non-diagonal part
    tvec = 0.0 * v
    for ith = 1:tDim-1
        coef = facs[ith]
        for (ind, jth) in enumerate(non0s[ith])
            fac = 0.5 * t * coef[ind]
            tvec[jth] += fac * v[ith]
            tvec[ith] += fac * v[jth]
        end
    end
    w += tvec
    return w
end

function Lanczos(t::Float64,N::Int64,non0s::AAI,
                 facs::AAF,Ufacs::FA,tDim::Int64,
                 numstate::Int64,lm=300) where{FA<:Array{Float64,1},
                                                     AAI<:Array{Array{Int64,1},1},
                                               AAF<:Array{Array{Float64,1},1}}
    vkm1 = zeros(Float64,tDim)
    vk   = zeros(Float64,tDim); vk[1]=1.0
    betakm1 = 0.0; betak=0.0; alphas = Float64[]; betas = Float64[]
    en = [1.e+10]; TF = [false]
    for k = 1:lm
        w = operateH(vk,t,non0s,facs,Ufacs,tDim)  - betak .* vkm1
        alphak = dot(vk,w) 
        push!(alphas,alphak)
        en = convcheck(k,numstate,tDim,alphas,betas,en,TF)
        if TF[1];println("En=",en[1:minimum([numstate,tDim])], " Eg.s/N ", en[1]/N, " Lanczos it=$k"); break; end
        if k==lm
            try;print_vec("En=",en[1:minimum([tDim,numstate])]);
            catch;print_vec("En",en);end
        end
        w .-= alphak .* vk 
        betakm1 = betak; betak = sqrt(dot(w,w)); push!(betas,betak)
        if betak == 0.0; println("betak=0.0");break;end
        vkm1 .= vk; vk .= w .* (1.0/betak)
    end
end

function convcheck(k::Int64,numstate::Int64,tDim::Int64,
                   alphas::FA,betas::FA,en::FA,
                   TF,
                   tol=1.e-14) where{FA<:Array{Float64,1}}
    if k < numstate
        return en
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
        if k == numstate; return nen; end      
        dif = maximum( abs.(en[1:iend] .- nen[1:iend]))
        if dif < tol
            TF[1] = true
            return nen
        else
            return nen
        end
    end
end

function samplerun()
    tJ = 1.0;S = 0.5;nums = 1;PBC = false;L = 4
    for L=4:2:16
        @time main(tJ,S,L,nums,PBC)
    end
end

function dmrg()
    tJ = 1.0;S = 0.5;nums = 1;PBC = false;L = 4
    println("\n\n")
    @time v_sys = main(tJ,S,L,nums,PBC;retbasis=true)
    println("\n\n")
end

samplerun()
dmrg()
