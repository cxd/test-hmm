namespace au.id.cxd.HMM.Matrix
open System
open System
open System.IO
open System.Text
open System.Collections.Generic
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open System.Runtime.Serialization
open System.Runtime.Serialization.Formatters.Binary
open au.id.cxd.HMM

(*
This version of the module uses the approach outlined in AIMA section 15.2

note the wikipedia article at:
http://en.wikipedia.org/wiki/Forward–backward_algorithm

is also consistent with the algorithms outlined in AIMA.
*)
module HiddenMarkovModel = 

    type Model = 
                { 
                       pi:float list; 
                       A:Matrix<float>; 
                       // this list is the collection of evidence state transition probabilties
                       // for each state variable (t) hence it's length is T
                       Bk:Matrix<float> list; 
                       states:string list; 
                       evidence:string list; 
                       epoch: int;
                       error: float;}


    type Prediction = { prob: float; state: string; evidence:string; t:int; success: bool; }


    (* retrieve the indices for supplied index *)
    let indices (modelEvidence:string list) (examples:string list) =
        List.map (fun (item:string) ->
                    List.tryFindIndex (fun (key:string) -> key.Equals(item, StringComparison.OrdinalIgnoreCase)) modelEvidence
                    ) examples
        |> List.filter (fun item -> 
                            match item with
                            | Some(n) -> true
                            | _ -> false)
        |> List.map (fun item -> Option.get item)


    (* make diagonal matrix 
    for time t in state j
    *)
    let makeDiag t (B:Matrix<float>) = 
        let column = B.Column(t)
        let Ot = DenseMatrix.init column.Count column.Count (
                 fun i j -> match i = j with
                            | true -> column.[i]
                            | false -> 0.0
                 )
        Ot

(*
The forward algorithm 

alphaScale is a normalising constant used to make probabilities sum to 1 for example 1 / sum(M)
T - is the state transition matrix for probability of transition from State i to State j = T_{ij} = P(S_j|S_i)
O - is the evidence/observation to state transition matrix. This is a set of matrices once for each state at time t.

Note O is a diagonal matrix containing probabilities for each evidence variable whilst in state t

So if B = [[0.1 0.9]; [0.8 0.2]] then for state 1 O_1 = [[0.1 0.0]; [0.0 0.8]]

so O_j = diag(B_0_j) that is it is the diagonal matrix formed from using column j of the B matrix. 

*)
    let forward endT (pi:float list) (T:Matrix<float>) (B:Matrix<float>) =
        (* locally scoped cache *)
        let map = new Dictionary<int, Matrix<float>>()
        let cache i methodFn = 
            match map.ContainsKey(i) with
            | true -> map.[i]
            | false -> 
                map.Add(i, methodFn())
                map.[i]

        let prior = DenseVector.init (List.length pi) (fun i -> pi.[i])
        let rec innerForward (t:int) (T:Matrix<float>) (B:Matrix<float>) =
            match t = 0 with
            | true -> 
                let O = makeDiag t B
                let temp = O * T.Transpose() * prior.ToColumnMatrix()
                let alpha = temp.RowSums().Sum()
                match alpha > 0.0 with
                | true -> temp * (1.0/ alpha)
                | false -> temp
            | false -> 
                let alphaT1 = cache (t-1) (fun () -> innerForward (t-1) T B)
                let O = makeDiag t B
                let temp = O * T.Transpose() * alphaT1
                let alpha = temp.RowSums().Sum()
                match alpha > 0.0 with
                | true -> temp * (1.0/ alpha)
                | false -> temp
        let O = makeDiag 0 B
        let t1 = match endT <= T.RowCount - 2 with
                 | true -> endT
                 | false -> T.RowCount - 2
        innerForward t1 T B



    (* 
    the backward equation as defined in AIMA section 15.2

    this will only be able to function with at 3 or more evidence variables
    *)
    let backward endT (pi:float list) (T:Matrix<float>) (B:Matrix<float>) = 
        let map = new Dictionary<int, Matrix<float>>()
        let cache i methodFn = 
            match map.ContainsKey(i) with
            | true -> map.[i]
            | false -> 
                map.Add(i, methodFn())
                map.[i]
        let prior = DenseVector.init (List.length pi) (fun i -> pi.[i])
        let len = match endT <= (T.RowCount) - 3 with 
                  | true -> endT
                  | false -> (T.RowCount - 3)
        let rec inner (t:int) (T:Matrix<float>) (B:Matrix<float>) =
            match (t = len) with
            | true -> 
                let O = makeDiag t B
                let temp = T * O * prior.ToColumnMatrix()
                let d = temp.ColumnSums().Sum()
                match d > 0.0 with
                | true -> temp / d
                | false -> temp
            | false -> 
                let O = makeDiag t B
                let backward1 = cache (t+2) (fun () -> inner (t+2) T B) 
                let temp = T * O * backward1
                let d = temp.ColumnSums().Sum()
                match d > 0.0 with
                | true -> temp / d
                | false -> temp
        inner 0 T B


    let gamma t i (pi:float list) (T:Matrix<float>) (B:Matrix<float>) =
        let f = forward t pi T B
        let b = backward t pi T B
        Console.WriteLine("Forward: {0}", f)
        Console.WriteLine("Back: {0}", b)
        // note f and b are a single column matrix (column vector)
        let g = f.[i,0] * b.[i,0]
        g

    
    let epsilon t i j (pi:float list) (T:Matrix<float>) (B:Matrix<float>) =
        let f = forward t pi T B
        let b = backward t pi T B
        Console.WriteLine("Pi: {0}", pi)
        Console.WriteLine("A: {0}", T)
        Console.WriteLine("B: {0}", B)
        Console.WriteLine("Forward: {0}", f)
        Console.WriteLine("Back: {0}", b)
        let temp = f.[i,0] * T.[i,j] * B.[t, j] * b.[j, 0]
        let c = f.ColumnSums().Sum()
        match c = 0.0 with
        | false -> (1.0 / c) * temp
        | true -> temp

     
    (*
    Given the model and the sequence of evidence variables
    compute the most likely state transition sequence.
        
$$
P(X_{t+k+1}|e_{1:t}) = \sum_{ x_{t+k} } P(X_{t+k+1} | x_{t+k}) P(x_{t+k}|e_{1:t}) 
$$


$$
\gamma_i(t) = P(x_t | V^t, \lambda)
$$
The equation to generate $\gamma_i$ is given in \cite{rab} and \cite{pc} as follows.
$$
\gamma_i(t) = \frac{\alpha_i(t)\beta_j(t)}{P(x_t | V^t, \lambda)} = \frac{\alpha_i(t)\beta_j(t)}{\sum_{i=1}^N \alpha_j(t)\beta_i(t)}
$$
Having obtained $\gamma_i(t)$ the most likely state $q_t$ emitted at time $t$ is determined by the state with the maximum probability for $1 \le i \le N$ \cite{rab}.
\begin{equation*}
q_t = \begin{array}{rl}
argmax_{1 \le i \le N}[\gamma_i(t)], & 1 \le t \le T.
 \end{array}
\end{equation*}


    *)
    let predict (model:Model) (evidenceSequence:string list) : Prediction list =
        let pi = model.pi
        let A = model.A
        let B = model.Bk
        let V = indices model.evidence evidenceSequence
        let T = (List.length model.states) - 1
        let stateCount = List.length model.states
        let evidenceCount = List.length model.evidence

        // find q_t = argmax_{1 \le j \le N} (gamma_j), 1 \le t \le T
        // that is the argmax of state j at evidence time T where T is the index of V
        List.map (fun v -> 
                    let t = V.[v]
                    
                    let O = match t <= (B.Length - 1) with 
                            | true -> B.[t]
                            | false -> B.[0]
                                            
                    let row = List.map (fun j -> 
                                            gamma j t pi A O) [0..(A.ColumnCount-1)]
                    
                    Console.WriteLine("Time: {0} Row: {1}", t, List.fold(fun (sb:StringBuilder) i -> sb.Append(i.ToString()).Append(" ")) (new StringBuilder()) row)
                    let max = Double.MinValue
                    let (max, index, searchIdx, flag) = 
                        List.fold(
                            fun (max, index, searchIdx, flag) p ->
                                match (p > max) with
                                | true -> (p, searchIdx, searchIdx+1, true)
                                | false -> (max, index, searchIdx+1, flag)
                                ) (max, 0, 0, false) row 
                    // map the pairs to a set of states
                    { prob = max; state = model.states.[index]; evidence = model.evidence.[t]; t = t; success = flag; } 
                    ) [0..((List.length V)-1)]



    (* 
    the training method makes use of the forward backward
    algorithm.

    input: the input model to start training
    trainSequences: the training sequences to present for learning, the last item in each sequence is considered to be the target state
    Note that training sequences need only be the set of complete sequences for each set of transitions.

    theta: the threshold to use until convergence
    maxEpochs: the maximum epochs to run if convergence is not met

    *)
    let train (input:Model) ( trainSequences:string list list) : Model = 
        let A = input.A
        let Bk = input.Bk
        let pi = input.pi
        let stateCount = List.length input.states 
        let evidenceCount = List.length input.evidence
        let states = input.states
        let totalSequences = List.length trainSequences
        let theta = input.error
        let maxEpochs = input.epoch

        // the goal is to learn a new matrix A and a new set of matrices O for each 
        // sequence at time T in the training sequences.



        // reestimation parameters
        (*
        restimating \bar{\pi}

        expected frequency in state i at time 1
        $$
        \bar{pi_i} = \gamma_1 (i)
        $$
        *)
        let piNew (V:int list) (lpi:float list) (lA:Matrix<float>) (lB:Matrix<float>)  =
            List.map (fun i -> gamma 1 i lpi lA lB) [0..(lA.ColumnCount-1)]
        
        (*
        restimating \bar{A}
        expected number of transitions from S_i to S_j / expected number of transitions from S_i
        $$
        \bar{a_{ij} } = \frac { \sum_{t=1}^{T} \epsilon_t(i, j) } {\sum_{t=1}^T \gamma_t(i)}
        $$
        *)
        let aNew (V:int list) (lpi:float list) (lA:Matrix<float>) (lB:Matrix<float> list) =
            let V2 = indices input.evidence input.evidence
            let denoms = List.map (
                            fun i ->
                                List.mapi (fun t Bt ->
                                            List.fold (fun n v -> n + gamma v i lpi lA Bt) 0.0 V2
                                            ) lB |> List.sum
                                            ) [0..(lA.RowCount - 1)]
            Console.WriteLine("denom: {0}",denoms)                    
            DenseMatrix.init lA.RowCount lA.ColumnCount (fun i j ->
                                let num = List.fold (fun n t ->
                                                let Bt = lB.[t]
                                                List.fold (fun n v -> n + epsilon v i j lpi lA Bt) 0.0 V
                                            ) 0.0 [0..((List.length lB) - 1)]
                                
                                Console.WriteLine("A eps: {0}", num)
                                match denoms.[i] = 0.0 with
                                | true -> 0.0
                                | false -> num / denoms.[i])
        
        (*
        restimating \bar{A}

        expected number of times in state j observing symbol v_k / expected number of times in state j 

        $$
        \bar{b_{ij}} = \frac{ \sum_{t=1}^T_k \gamma_t(j) } {\sum_{t=1}^T gamma_t(j) }
        $$
        $$
        s.t. O_{Tk} = v_k 
        $$
        the observation at time Tk contains the symbol v_k
        *)
        let bNew (V:int list) (lpi:float list) (lA:Matrix<float>) (lB:Matrix<float> list) =
            let V2 = indices input.evidence input.evidence
            let denom = List.map(fun i ->
                            List.mapi (fun t Bt ->
                                    // gamma state i
                                    List.fold (fun n v -> n + gamma v i lpi lA Bt) 0.0 V2
                                    ) lB |> List.sum) [0..(lA.RowCount - 1)] 
                        |> List.sum
            Console.WriteLine("B denoms: {0}", denom)            
            List.mapi (fun t (Bt:Matrix<float>) ->
                        Console.WriteLine("t: {0}", t)
                        Console.WriteLine("Bt: {0}", Bt)
                        DenseMatrix.init Bt.RowCount lA.ColumnCount (fun i j ->
                            // gamma state j
                            let num = gamma j i lpi lA Bt
                            Console.WriteLine("Gamma: {0}", num)
                            match denom = 0.0 with
                            | true -> 0.0
                            | false -> num / denom
                        )) lB
            
        
        let rec innerTrain epoch error (piBar:float list) (barA:Matrix<float>) (barB:Matrix<float> list) =
            match epoch >= maxEpochs with
            | true -> (epoch, error, piBar, barA, barB)
            | false ->
                let (oldA, oldB) = (barA, barB)
                let (pi2, Anew, Bnew) = 
                    List.fold (fun ((pi, barA, barB):float list * Matrix<float> * Matrix<float> list) trainSequence ->
                                match (List.length trainSequence) > 2 with
                                | false -> (pi, barA, barB)
                                | true ->
                                    let V = indices input.evidence trainSequence
                                    let pi2 = piNew V pi barA barB.[0]
                                    let Anew = aNew V pi2 barA barB
                                    let Bnew = bNew V pi2 barA barB
                                    Console.WriteLine("Pi2: {0}", pi2)
                                    Console.WriteLine("A NEW: {0}", Anew)
                                    Console.WriteLine("B NEW: {0}", Bnew)
                                    (pi2, Anew, Bnew)) (piBar, barA, barB) trainSequences
                let matrixDelta (M1:Matrix<float>) (M2:Matrix<float>) = 
                     Array.zip (M1.ToColumnWiseArray()) (M2.ToColumnWiseArray())
                             |> (fun pairs -> Array.map (fun ((a1, a2):float * float) -> Math.Abs(Math.Abs(a1) - Math.Abs(a2))) pairs)
                             |> Array.sum

                // work out the deltas for A
                let deltaA = matrixDelta oldA Anew
                let deltaB = List.zip Bnew oldB 
                             |> (fun pairs -> 
                                    List.fold (fun n (B1, B2) ->
                                            n + matrixDelta B1 B2
                                            ) 0.0 pairs)
                let maxErr = List.max [deltaA; deltaB]
                Console.WriteLine("Epoch: {0} Error: {1}", epoch, maxErr)
                match (maxErr < theta) with
                | true -> (epoch, maxErr, pi2, Anew, Bnew)
                | false -> innerTrain (epoch+1) maxErr pi2 Anew Bnew

        // train sequences
        let (epoch, error, newPi, newA, newB) = innerTrain 1 0.0 pi A Bk
        { pi = newPi; 
          A = newA; 
          Bk = newB; 
          states = input.states; 
          evidence = input.evidence; 
          epoch = epoch;
          error = error; }


