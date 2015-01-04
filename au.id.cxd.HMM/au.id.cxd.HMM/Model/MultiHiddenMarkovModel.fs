namespace au.id.cxd.HMM.Multi

open System
open System.IO
open System.Text
open System.Collections.Generic
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open System.Runtime.Serialization
open System.Runtime.Serialization.Formatters.Binary
open au.id.cxd.HMM

module MultiHiddenMarkovModel =


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
        List.iter (fun (idx:string) -> Console.WriteLine("Find Index: {0}", idx)) examples 
        List.map (fun (item:string) ->
                    List.tryFindIndex (fun (key:string) -> key.Equals(item, StringComparison.OrdinalIgnoreCase)) modelEvidence
                    ) examples
        |> List.filter (fun item -> 
                            match item with
                            | Some(n) -> true
                            | _ -> false)
        |> List.map (fun item -> Option.get item)

    (* forward equation
   
    T = time T
    pi = P(x_i)
    a_ij = P(x_j | x_i)
   b_ij = P(x_j | e_i)

   T represents i current state we need to predict j next state

   V - indices of evidence variables observed up to time T
    *)
    let rec alpha (T:int) (pi:float list) (A:Matrix<float>) (B:Matrix<float>) (stateCount:int) (evidenceCount:int) (V:int list):Matrix<float> = 
        (* locally scoped cache *)
        let map = new Dictionary<int, Matrix<float>>()
        let cache i methodFn = 
            match map.ContainsKey(i) with
            | true -> map.[i]
            | false -> 
                map.Add(i, methodFn())
                map.[i]

        (* recurse on 
        time T 
        with priors pi 
        and state transition matrix A 
        with evidence matrix B
        and index of transitions V
        accumulate in matrix accum
        *)
        let rec alphaInner (T:int) (pi:float list) (A:Matrix<float>) (B:Matrix<float>) (V:int list) (accum:Matrix<float>):Matrix<float> =
            // compute new probabilities for evidence var at vk, at time t and update the row t in matrix accum
            (*
            \alpha_j(t) = b_{jk}v(t)\sum_{i=1}^c \alpha_i(t-1)a_{ij}

            vk = current evidence variable in V
            t = index of state at time t

            *)
            let inner (vk:int) (t:int) (accum:Matrix<float>):Matrix<float> =
                //Console.WriteLine("Accum: {0}", accum)

                let alpha2 = cache (t-1) (fun () -> alphaInner (t - 1) pi A B V accum)


                let totals = DenseMatrix.init accum.RowCount accum.ColumnCount (
                                    fun i j ->
                                        alpha2.[i, t - 1] * A.[t, j] )
                
                let sums = totals.ColumnSums()
                let total = DenseVector.init totals.ColumnCount (
                                fun j -> sums.[j] * B.[vk, j]
                            )
                
                accum.SetRow(vk, total)
                accum

            match T = 0 with
            | true -> accum
            | _ -> List.fold (fun (accum:Matrix<float>) (k:int) -> 
                                let alpha1 = inner k T accum
                                // scaling \hat{\alpha_t} = \frac{1}{\sum_i \bar{\alpha_t}(i)
                                let scale = alpha1.ColumnSums().Sum()
                                let alpha2 = match scale = 0.0 with
                                             | true -> alpha1
                                             | false -> alpha1 / scale
                                alpha2
                                ) accum V

        let accum = DenseMatrix.init evidenceCount stateCount (
                        fun t j ->
                            let vlen = List.length V
                            match t < vlen with
                            | true -> let idx = V.[t]
                                      pi.[j] * B.[idx, j]
                            | false -> pi.[j] // (Convert.ToDouble(evidenceCount))
                    )
        let accum' = DenseMatrix.init accum.RowCount accum.ColumnCount (
                        fun i j ->
                            let row = accum.Row(i)
                            let c = row.Sum()
                            match c = 0.0 with
                            | false -> accum.[i,j]/c
                            | _ -> accum.[i,j]
                    )

        let alphaResult = alphaInner T pi A B V accum'
        map.Clear()
        alphaResult.NormalizeRows(1.0)


     (* 

     the unscaled alpha forward equation used during the beta backward equation
   
    T = time T
    pi = P(x_i)
    a_ij = P(x_j | x_i)
   b_ij = P(x_j | e_i)

   T represents i current state we need to predict j next state

   V - indices of evidence variables observed up to time T

   Based on the Erratum for "A Tutorial on Hidden Markov Models..."
   the scale factor for \hat{\alpha_t} = c_t \bar{\alpha_t}
   where 
   \bar{\alpha_t}(j) = \sum_i^N \hat{\alpha_{t-1}}(i)a_{ij}b_j(O_{t}

   and

   c_t = \frac{1}{sum_i \bar{\alpha_t}(i)}

   in this case the last iteration is not scaled with the scaling factor c_{t}
    *)
    let rec alphaUnscaled (T:int) (pi:float list) (A:Matrix<float>) (B:Matrix<float>) (stateCount:int) (evidenceCount:int) (V:int list):Matrix<float> = 
        let finalT = T
        (* locally scoped cache *)
        let map = new Dictionary<int, Matrix<float>>()
        let cache i methodFn = 
            match map.ContainsKey(i) with
            | true -> map.[i]
            | false -> 
                map.Add(i, methodFn())
                map.[i]

        (* recurse on 
        time T 
        with priors pi 
        and state transition matrix A 
        with evidence matrix B
        and index of transitions V
        accumulate in matrix accum
        *)
        let rec alphaInner (T:int) (pi:float list) (A:Matrix<float>) (B:Matrix<float>) (V:int list) (accum:Matrix<float>):Matrix<float> =
            // compute new probabilities for evidence var at vk, at time t and update the row t in matrix accum
            (*
            \alpha_j(t) = b_{jk}v(t)\sum_{i=1}^c \alpha_i(t-1)a_{ij}

            vk = current evidence variable in V
            t = index of state at time t

            *)
            let inner (vk:int) (t:int) (accum:Matrix<float>):Matrix<float> =
                //Console.WriteLine("Accum: {0}", accum)

                let alpha2 = cache (t-1) (fun () -> alphaInner (t - 1) pi A B V accum)


                let totals = DenseMatrix.init accum.RowCount accum.ColumnCount (
                                    fun i j ->
                                        alpha2.[i, t - 1] * A.[t, j] )
                
                let sums = totals.ColumnSums()
                let total = DenseVector.init totals.ColumnCount (
                                fun j -> sums.[j] * B.[vk, j]
                            )
                
                accum.SetRow(vk, total)
                accum

            match T = 0 with
            | true -> accum
            | _ -> List.fold (fun (accum:Matrix<float>) (k:int) -> 
                                let alpha1 = inner k T accum
                                match T = finalT with
                                | true -> 
                                    // the unscaled equation will not scale the last iteration resulting in \bar{\alpha} rather than \hat{\alpha}
                                    alpha1
                                | false -> 
                                    // scaling \hat{\alpha_t} = \frac{1}{\sum_i \bar{\alpha_t}(i)
                                    let scale = alpha1.ColumnSums().Sum()
                                    let alpha2 = match scale = 0.0 with
                                                 | true -> alpha1
                                                 | false -> alpha1 / scale
                                    alpha2
                                ) accum V

        let accum = DenseMatrix.init evidenceCount stateCount (
                        fun t j ->
                            let vlen = List.length V
                            match t < vlen with
                            | true -> let idx = V.[t]
                                      pi.[j] * B.[idx, j]
                            | false -> pi.[j] // (Convert.ToDouble(evidenceCount))
                    )
        let accum' = DenseMatrix.init accum.RowCount accum.ColumnCount (
                        fun i j ->
                            let row = accum.Row(i)
                            let c = row.Sum()
                            match c = 0.0 with
                            | false -> accum.[i,j]/c
                            | _ -> accum.[i,j]
                    )

        let alphaResult = alphaInner T pi A B V accum'
        map.Clear()
        alphaResult.NormalizeRows(1.0)

    (*
    this is the time reversed algorithm of alpha
    moving from t = 1 to T

    T = time T
    pi = P(x_i)
    a_ij = P(x_j | x_i)
   b_ij = P(x_j | e_i)

   T represents i current state we need to predict j next state

   V - indices of evidence variables observed up to time T
    *)
    let beta (T:int) (pi:float list) (A:Matrix<float>) (B:Matrix<float>) (stateCount:int) (evidenceCount:int) (V:int list):Matrix<float> = 
        (* locally scoped cache *)
        let map = new Dictionary<int, Matrix<float>>()
        let cache i methodFn = 
            match map.ContainsKey(i) with
            | true -> map.[i]
            | false -> 
                map.Add(i, methodFn())
                map.[i]

        let alpha0 = alphaUnscaled T pi A B stateCount evidenceCount V

        let rec betaInner (T:int) (t:int) (pi:float list) (A:Matrix<float>) (B:Matrix<float>) (V:int list) (accum:Matrix<float>) =
            // perform the calculation 
            (*

            \beta_i(t) = \sum_{j=1}^C \beta_j(t+1)a_{ij}b_{jk}v(t+1)
            *)
            let inner (vk:int) (t:int) (accum:Matrix<float>):Matrix<float> =
                let beta2 = cache (t+1) (fun () -> betaInner T (t + 1) pi A B V accum)
                // alpha2.[i, t - 1] * A.[t-1, j] * B.[vk, t]
                // calculate the scaling factor D_t = \prod_{\t=1}^T c_t

                let d = Array.reduce (fun a b -> a+b) (alpha0.ToColumnWiseArray())
                // vk = i, t
                // \beta_i(t) = \sum_{j=1}^C \beta_j(t+1)a_{ij}b_{jk}v(t+1)
                let totals = DenseMatrix.init accum.RowCount accum.ColumnCount (
                                    fun i j ->
                                        beta2.[i, t + 1] * A.[t, j] * B.[vk, t + 1]) 
                let total = totals.ColumnSums()
                // scaling coefficient
                match d = 0.0 with
                | false ->
                    // \beta_t(i)
                    accum.SetRow(vk, total / d)
                    accum
                | true -> accum
            
            match t >= (T-1) with
            | true -> accum
            | false -> List.fold (fun (accum:Matrix<float>) (k:int) -> inner k (t+1) accum) accum V

        // initialising beta matrix with priors
        let accum = DenseMatrix.init evidenceCount stateCount (
                        fun t j ->
                            let vlen = List.length V
                            match t < vlen with
                            | true -> let idx = V.[t]
                                      pi.[j] * B.[idx, j]
                            | false -> pi.[j] / (Convert.ToDouble(evidenceCount))
                    )

        let accum' = DenseMatrix.init accum.RowCount accum.ColumnCount (
                        fun i j ->
                            let row = accum.Row(i)
                            let c = row.Sum()
                            match c = 0.0 with
                            | false -> accum.[i,j]/c
                            | _ -> accum.[i,j]
                    )
        
        //let test = DenseMatrix.init (List.length V) A.ColumnCount ( fun i j -> 1.0 )
        let betaResult = betaInner T 0 pi A B V accum'
        map.Clear()
        betaResult.NormalizeRows(1.0)


(* we use the baum welch algorithm which defines gamma and epsilon
        alphaM is a matrix of MxN size where a_ij = P(e_i | S_j)
        betaM is a matrix of MxN size where b_ij = P(e_i | S_j)
        gamma = P(q_t = S_i | O, \lambda)
        the probability of being in state S_i at time t given the sequence O
        $$
        \gamma_t(i) = \frac{\alpha_t(i) \beta_t(i)}{\sum_{j=1}^N \alpha_t(j)\beta_t(j) } 
        $$
        gamma matrix can be constructed as a t x m matrix for time t
        note the value of O is defined by the supplied alpha and beta matrices

        note alphaM and betaM are considered to be constructed at time t.

        *)
    let gamma i t (alphaM:Matrix<float>) (betaM:Matrix<float>) : float =
            Console.WriteLine("Alpha: {0}", alphaM)
            Console.WriteLine("Beta: {0}", betaM)
            let denom = List.map (fun j ->
                                    alphaM.[t, j]*betaM.[t, j]) [0..(alphaM.ColumnCount - 1)]
                        |> List.sum 
            let num = alphaM.[t, i] * betaM.[t,i]
            let g = match denom = 0.0 with
                    | true -> 0.0
                    | false -> num / denom
            Console.WriteLine("Gamma: {0} T: {1} G: {2}", i, t, g)
            g

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
                    let row = List.map (fun j -> 
                                            let O = B.[t]
                                            let alphaM = alpha T pi A O stateCount evidenceCount V
                                            let betaM = beta T pi A O stateCount evidenceCount V
                                            gamma j t alphaM betaM) [0..(A.ColumnCount-1)]
                    
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



        (*
        The epsilon function determines the probability of transitioning from S_i at t to S_j at t+1 for the current the observation sequence given the model.
        $$
        \epsilon_t(i, j) = P(q_t = S_i, q_{t+1}=S_j, O | \lambda)
        $$
        $$
        = \frac{\alpha_t(i)a_{ij}b_j(O_{t+1})\beta_{t+1}(j)}{\sum_{i=1}\sum_{j=1}\alpha_t(i)\a_{ij}b_j(O_{t+1})\beta_{t+1}(j)}
        $$

        Note the input matrix for B and betaM must be computed for time t+1 prior to supplying to this method.

        *)
        let epsilon t i j (alphaM:Matrix<float>) (betaMT1:Matrix<float>) (A:Matrix<float>) (BT1:Matrix<float>) : float =
            // compute the denominator
            let denom = List.fold (fun n i1 ->
                                    n + (List.map (fun j1 -> 
                                                alphaM.[t, i1] * A.[i1, j1] * BT1.[t, j] * betaMT1.[t, j]
                                        ) [0..(A.ColumnCount - 1)] |> List.sum) ) 0.0 [0..(A.RowCount-1)]
            let num = alphaM.[t, i] * A.[i, j] * BT1.[t, j] * betaMT1.[t, j]
            match denom = 0.0 with
            | true -> 0.0
            | false -> num / denom

        // reestimation parameters
        (*
        restimating \bar{\pi}

        expected frequency in state i at time 1
        $$
        \bar{pi_i} = \gamma_1 (i)
        $$
        *)
        let piNew (V:int list) (lpi:float list) (lA:Matrix<float>) (lB:Matrix<float>)  =
            let alphaM = alpha ((List.length states)-1) lpi lA lB stateCount evidenceCount V   
            let betaM = beta ((List.length states) - 1) lpi lA lB stateCount evidenceCount V 
            List.map (fun i -> gamma 0 i alphaM betaM) [0..(alphaM.ColumnCount-1)]
        
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
                                            let alphaM = alpha ((List.length states)-1) lpi lA Bt stateCount evidenceCount V   
                                            let betaM = beta ((List.length states) - 1) lpi lA Bt stateCount evidenceCount V
                                            List.fold (fun n v -> n + gamma i v alphaM betaM) 0.0 V2
                                            ) lB |> List.sum
                                            ) [0..(lA.RowCount - 1)]
            Console.WriteLine("denom: {0}",denoms)                    
            DenseMatrix.init lA.RowCount lA.ColumnCount (fun i j ->
                                let num = List.fold (fun n t ->
                                                let Bt = lB.[t]
                                                let Bt1 = lB.[t+1]
                                                let alphaM = alpha ((List.length states)-1) lpi lA Bt stateCount evidenceCount V   
                                                let betaM = beta ((List.length states) - 1) lpi lA Bt1 stateCount evidenceCount V 
                                                List.fold (fun n v -> n + epsilon v i j alphaM betaM lA Bt) 0.0 V
                                            ) 0.0 [0..((List.length lB) - 2)]
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
                                    let alphaM = alpha ((List.length states)-1) lpi lA Bt stateCount evidenceCount V   
                                    let betaM = beta ((List.length states) - 1) lpi lA Bt stateCount evidenceCount V
                                    // gamma state i
                                    List.fold (fun n v -> n + gamma i v alphaM betaM) 0.0 V2
                                    ) lB |> List.sum) [0..(lA.RowCount - 1)] 
                        |> List.sum
            Console.WriteLine("B denoms: {0}", denom)            
            List.mapi (fun t (Bt:Matrix<float>) ->
                        let alphaM = alpha ((List.length states)-1) lpi lA Bt stateCount evidenceCount V   
                        let betaM = beta ((List.length states) - 1) lpi lA Bt stateCount evidenceCount V
                        Console.WriteLine("t: {0}", t)
                        Console.WriteLine("Bt: {0}", Bt)
                        Console.WriteLine("Alpha: {0}", alphaM)
                        Console.WriteLine("Beta: {0}", betaM)
                        DenseMatrix.init Bt.RowCount lA.ColumnCount (fun i j ->
                            // gamma state j
                            let num = gamma j i alphaM betaM
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

 