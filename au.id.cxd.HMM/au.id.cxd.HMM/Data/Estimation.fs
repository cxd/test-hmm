namespace au.id.cxd.HMM

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra

module Estimation = 
    (*
    extract all sequences that end in the supplied state.
    *)
    let extractStateSeq (data : string list list) (state : string) = 
        List.fold (fun (accum : string list list) (items : string list) -> 
            let head = List.rev items |> List.head
            match (head.ToLower().Equals(state.ToLower())) with
            | true -> items :: accum
            | false -> accum) [] data
        |> List.rev

    (*
    extract all sequences whose last "term" (the item before the state) is equal to the end term.
    *)
    let extractEndTermSeq (data: string list list) (endTerm: string) =
        List.fold (
            fun (accum : string list list) (items : string list) ->
                let last = List.rev items |> List.tail |> List.head
                match (last.ToLower().Equals(endTerm.ToLower())) with
                | true -> items :: accum
                | false -> accum
            ) [] data
    
    (* for each state in states count the frequency and return a vector 
    this will add 1 to every state so there are no 0 values
    *)
    let countStateFreq (data : string list list) (states : string list) = 
        List.map (fun (state : string) -> 
            List.fold (fun ((n, state) : float * string) (items : string list) -> 
                let head = List.rev items |> List.head
                match (head.ToLower().Equals(state.ToLower())) with
                | true -> (n + 1.0, state)
                | false -> (n, state)) (1.0, state) data) states
    
    (* calculate the priors for the states parameter pi *)
    let statePriors (data : string list list) (states : string list) = 
        let frequencies = countStateFreq data states
        let sum = List.sumBy (fun (n, state) -> n) frequencies
        let priors = List.map (fun (n, state) -> n / sum) frequencies
        priors
    
    let matches (a : string) (b : string) = a.Equals(b, StringComparison.OrdinalIgnoreCase)

    (*
    count the number of items different between lists
    listA should be length = listB length
    *)
    let countDelta (listA : string list) (listB : string list) = 
        let len = List.length listA

        let test = List.toSeq listB |> Seq.take len |> Seq.toList
        
        let pairs = List.zip listA test
        List.fold (fun n (itemA, itemB) -> 
            match (matches itemA itemB) with
            | true -> n
            | false -> n + 1.0) 0.0 pairs

    (*
    count the number of items different between lists
    listA should be length <= listB length
    This includes the state class and removes it
    *)
    let countDeltaState (listA : string list) (listB : string list) = 
        let arr = List.toArray listA
        
        let test = 
            List.mapi (fun i item -> 
                if i >= (Array.length arr) - 1 then ""
                else arr.[i]) listB
        
        let pairs = List.zip test listB
        List.fold (fun n (itemA, itemB) -> 
            match (matches itemA itemB) with
            | true -> n
            | false -> n + 1.0) 0.0 pairs
    
    (* count transitions from stateA to stateB 
    this will always add 1 to every transition so there is no 0 values
    *)
    let countTransitionFreq (data : string list list) (stateA : string) (stateB : string) = 
        let findSeq (state : string) = 
            List.fold (fun accum (items : string list) -> 
                let head = List.rev items |> List.head
                match (head.ToLower().Equals(state.ToLower())) with
                | true -> items :: accum
                | false -> accum)
        
        // find all sequences that end in stateA
        let startSeqs = findSeq stateA [] data
        // find all sequences that end in stateB
        let endSeqs = findSeq stateB [] data
        
        // for each start sequence 
        let freqs = 
            List.map (fun startSeq -> 
                let otherSeqs = List.filter (fun others -> (List.length others) = (List.length startSeq) + 1) endSeqs
                
                let sums = 
                    List.fold (fun n others -> 
                        let delta = countDeltaState startSeq others
                        match (delta > 2.0) with
                        | true -> n
                        | false -> n + 1.0) 1.0 otherSeqs
                sums) startSeqs
        match (List.length freqs) = 0 with
        | true -> 1.0
        | false -> List.sum freqs
    
    (*
    count transitions between states

    from stateA (i) to stateB (j)

    *)
    let countTransitions (data : string list list) (states : string list) = 
        let arr = List.toArray states
        
        let mat = 
            DenseMatrix.init (List.length states) (List.length states) (fun i j -> 
                let stateA = arr.[i]
                let stateB = arr.[j]
                countTransitionFreq data stateA stateB)
        mat
    
    (*
    determine the probability for the state transitions

    from stateA (i) to stateB (j)

    a_ij = P(x_j | x_i)

    Note each row will sum to 1.

    *)
    let stateTransitions (data : string list list) (states : string list) = 
        let mat = countTransitions data states
        mat.NormalizeRows(1.0)

    (* extract all unique sequences that end with the supplied end state
    each unique sequence will eventually result in a matrix B_k of P(e_j | x_i)
    *)
    let uniqueSequences (data : string list list) (endState : string) = 
        let sequences = extractStateSeq data endState
        // determine whether any of these are duplicates
        let text = 
            List.map 
                (fun (items : string list) -> List.fold (fun (accum : string) (item : string) -> accum + item) "" items) 
                sequences
        List.zip text sequences
        |> List.toSeq
        |> Seq.distinctBy (fun (txt : string, items : string list) -> txt)
        |> Seq.map (fun (txt : string, items : string list) -> items)
        |> Seq.toList
    
    (* extract all subsequences for a given end sequence 
    ignore any sequence that is longer than endSeq length
    *)
    let subsequences (data : string list list) (endSeq : string list) = 
        let subset = List.filter (fun (items : string list) -> List.length items <= List.length endSeq) data
        List.fold (fun (accum : string list list) (items : string list) -> 
            // work out if the sequence items is a subset of endSeq
            let len = List.length items
            
            let seqA = 
                items
                |> List.toSeq
                |> Seq.take (len - 1)
                |> Seq.toList
            
            //Console.WriteLine("A = {0}", seqA)

            let seqB = 
                endSeq
                |> List.toSeq
                |> Seq.take (len - 1)
                |> Seq.toList

            //Console.WriteLine("B = {0}", seqB) 
            
            let delta = countDelta seqA seqB

            //Console.WriteLine("Delta = {0}", delta)

            match (delta) with
            | 0.0 -> items::accum
            | _ -> accum) [] subset

    (* extract all subsequences for the supplied end sequence *)
    let allSubsequences (data:string list list) (endSequences: string list list) =
        List.map (fun (endSeq:string list) -> subsequences data endSeq) endSequences

    (* count the occurance of the term in the sequence
    this will always add 1.0 to every term count so there are no 0 values.
    *)
    let countTermInSeq (items:string list) (term:string) =
        List.fold(fun n (item:string) -> 
            match (matches term item) with 
            | true -> n + 1.0
            | false -> n) 1.0 items

    (* determine whether the end term matches the supplied term from the sequence

    The sequence is assumed to contain

    [term1; term2; ...; termN; state]

    This will reverse the term list and remove the state
    it will then determine whether termN matches the supplied term.

    *)
    let countTermInEndOfSeq (items:string list) (term:string) =
        let rev = List.rev items
        let state = List.head rev
        let endTerm = List.head (List.tail rev)
        match (matches term endTerm) with 
            | true -> 2.0
            | false -> 1.0


    (* count the frequency of the term in the sequence 
    return a tuple (double * double)
    where _1 = term frequency for supplied state
    _2 = total term frequency in subseq

    This method will always start with 1.0 adding 1.0 to every term count
    so that there is no 0.0 values

    *)
    let countTermInState (subseq:string list list) (term:string) (state:string) =
        let stateSeq = extractStateSeq subseq state
        let stateCount = List.fold(fun n (items:string list) ->
                                n + (countTermInSeq items term)
                                ) 1.0 stateSeq
        let totalCount = List.fold(fun n (items:string list) ->
                                n + (countTermInSeq items term)
                            ) 1.0 subseq
        (stateCount, totalCount)

    (* 
    this will count the total number of times a term appears at the end of a sequence
    for the supplied state label

    It will then count the total number of occurances of the term in all sequences

    and will return the tuple

    (endOfSeqCount, totalCount)
    *)
    let countTermAtEndOfState (data:string list list) (term:string) (state:string) =
        let stateSeq = extractStateSeq data state
        let stateCount = List.fold(fun n (items:string list) ->
                                n + (countTermInEndOfSeq items term)
                                ) 1.0 stateSeq
        let totalCount = List.fold(fun n (items:string list) ->
                                n + (countTermInSeq items term)
                            ) 1.0 data
        (stateCount, totalCount)

    (* determine the prior evidence matrix P(e_i | x_j) 
    for state transitions

    each row will normalise to 1.

    *)
    let priorEvidence (data:string list list) (terms:string list) (states:string list) =
        let m = List.length states
        let n = List.length terms
        let stateArr = List.toArray states
        let termArr = List.toArray terms
        let mat = DenseMatrix.init n m (fun i j -> 
            // term is i, state is j P(e_i | x_j)
            let (freq, totalFreq) = countTermInState data termArr.[i] stateArr.[j]
            freq / totalFreq
            )
        mat.NormalizeRows(1.0)

    (* calculate the prior evidences for all data sets *)
    let priorEvidences (dataSet:string list list) (terms:string list) (states:string list) =
        List.map (fun term -> 
                      // extract all sequences from the data set that end in the current term.
                      let data = extractEndTermSeq dataSet term
                      // generate a state transition matrix for the subset.
                      priorEvidence data terms states
                      ) terms

  

  
     (* 
     calculate the prior evidences for all data sets
        assume each sample is independently and identically distributed
        return the average prior evidences
     *)

    let avgPriorEvidences (dataSet:string list list) (terms:string list) (states:string list) =
        let m = List.length states
        let n = List.length terms
            
        let innerPriorEvidence (data:string list list) (terms:string list) (states:string list) =
            let stateArr = List.toArray states
            let termArr = List.toArray terms
            let total = ref 0.0

            let result = List.fold(
                            fun accum i ->
                                let arr = 
                                    List.fold(
                                        fun arr j ->
                                    
                                            let (freq, totalFreq) = countTermAtEndOfState data termArr.[i] stateArr.[j]

                                            ((i,j), (freq, totalFreq)) :: arr) accum [0..(m-1)]  
                                arr) [] [0..(n-1)]

            Console.WriteLine("RESULT:")
            List.iter (fun ((i,j), (f, t)) -> 
                        Console.WriteLine("[{0},{1}] = [{2},{3}]", i, j, f, t)) result


            let totalVec = DenseVector.init n (
                            fun i ->
                                List.fold(fun n ((a,b), (f, t)) ->
                                    match (i = a) with
                                    | true -> n + t
                                    | false -> n
                                    ) 0.0 result)

            let mat = DenseMatrix.init n m (fun i j -> 
                // term is i, state is j P(e_i | x_j)
                let total = totalVec.[i]
                let freq = List.fold(fun n ((a, b), (f, t)) ->
                                        match (i = a) && (j = b) with
                                        | true -> f
                                        | false -> n
                                    ) 0.0 result
                freq
                )
            (totalVec, mat)

        
        let B = DenseMatrix.init n m (fun i j -> 0.0)
        let V = DenseVector.init n (fun i -> 0.0)

        let (V, B) = innerPriorEvidence dataSet terms states
        Console.WriteLine("Totals:{0}{1}", Environment.NewLine, V)
        Console.WriteLine("MAT:{0}{1}", Environment.NewLine, B)

        (DenseMatrix.init n m (fun i j -> 
                                let t = V.[i]
                                B.[i,j]/t)).NormalizeRows(1.0)


 


    (*
    Compute the set of transition matrices between S_i and S_j at time t when evidence variable t is observed.
    This results in a list of matrices for each evidence variable (term) of state transitions between S_i and S_j
    *)
    let evidenceTransition (dataSet:string list list) (terms:string list) (states:string list) =
        List.map (fun term -> 
                      // extract all sequences from the data set that end in the current term.
                      let data = extractEndTermSeq dataSet term
                      // generate a state transition matrix for the subset.
                      stateTransitions data states
                      ) terms