namespace au.id.cxd.HMM

open System
open System.IO
open System.Text
open System.Collections.Generic
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open System.Runtime.Serialization
open System.Runtime.Serialization.Formatters.Binary
open au.id.cxd.HMM.DataTypes

module HiddenMarkovModel =

    type InputModel = {pi:float list;
                       A:Matrix<float>;
                       Bk:Matrix<float> list;
                       states:string list;
                       evidence:string list;}

    type Model = { pi:float list; 
                   A:Matrix<float>; 
                   B:Matrix<float>; 
                   states:string list; 
                   evidence:string list; 
                   epoch: int;
                   error: float;}                 

    
    type Prediction = { prob: float; state: string; evidence:string; t:int; success: bool; }

    (* retrieve the indices for supplied index *)
    let indices (modelEvidence:string list) (examples:string list) =
        List.iter (fun (idx:string) -> Console.WriteLine("Find Index: {0}", idx)) examples 
        List.map (fun (item:string) ->
                    List.findIndex (fun (key:string) -> key.Equals(item, StringComparison.OrdinalIgnoreCase)) modelEvidence
                    ) examples

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
                                        alpha2.[i, t - 1] * A.[t-1, j] )
                
                let total = totals.ColumnSums() * B.[vk, t]
                // scaling coeff
                (*
                c_{t+1} = \sum_{i=1}^N \hat{\alpha_t|(i) a_{ij} b_j(O_{t+1})
                *)
                let c = total.Sum()
                match c = 0.0 with
                | false -> 
                    //Console.WriteLine("Set Row: {0} Dim: [{1}, {2}]", vk, accum.RowCount, accum.ColumnCount)
                    //Console.WriteLine("Row Data: {0}", total/c)
                    accum.SetRow(vk, total / c)
                    accum
                | true -> accum

            match T = 0 with
            | true -> accum
            | _ -> List.fold (fun (accum:Matrix<float>) (k:int) -> inner k T accum) accum V

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

        Console.WriteLine("Start Dim: [{0}, {1}]", accum.RowCount, accum.ColumnCount)
        Console.WriteLine("Start: {0}{1}", Environment.NewLine, accum')        
            
        let alphaResult = alphaInner T pi A B V accum'
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

        let alpha0 = alpha T pi A B stateCount evidenceCount V

        let rec betaInner (T:int) (t:int) (pi:float list) (A:Matrix<float>) (B:Matrix<float>) (V:int list) (accum:Matrix<float>) =
            // perform the calculation 
            (*

            \beta_i(t) = \sum_{j=1}^C \beta_j(t+1)a_{ij}b_{jk}v(t+1)
            *)
            let inner (vk:int) (t:int) (accum:Matrix<float>):Matrix<float> =
                let beta2 = cache (t+1) (fun () -> betaInner T (t + 1) pi A B V accum)
                // alpha2.[i, t - 1] * A.[t-1, j] * B.[vk, t]
                // calculate the scaling factor D_t = \prod_{\t=1}^T c_t

                let alphaCol = alpha0.Column(t)
                let d = Array.reduce (fun a b -> a+b) (alphaCol.ToArray())


                // \beta_i(t) = \sum_{j=1}^C \beta_j(t+1)a_{ij}b_{jk}v(t+1)
                let totals = DenseMatrix.init accum.RowCount accum.ColumnCount (
                                    fun i j ->
                                        beta2.[i, t + 1] * A.[t, j] * B.[vk, t + 1]) 
                let total = totals.ColumnSums()
                // scaling coefficient
                match d = 0.0 with
                | false ->
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
        
        Console.WriteLine("Start Dim: [{0}, {1}]", accum.RowCount, accum.ColumnCount)
        Console.WriteLine("Start: {0}{1}", Environment.NewLine, accum') 

        //let test = DenseMatrix.init (List.length V) A.ColumnCount ( fun i j -> 1.0 )
        let betaResult = betaInner T 0 pi A B V accum'
        betaResult.NormalizeRows(1.0)

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
    let predict (model:Model) (evidenceSequence:string list) =
        let pi = model.pi
        let A = model.A
        let B = model.B
        let V = indices model.evidence evidenceSequence
        let T = (List.length model.states) - 1
        let stateCount = List.length model.states
        let evidenceCount = List.length model.evidence
        let alphaM = alpha T pi A B stateCount evidenceCount V
        let betaM = beta T pi A B stateCount evidenceCount V

        (* calculate the gamma function at state i in time t *)
        let gammaFn j t t_sub1 =
            let total = List.fold(fun n i ->
                                n + alphaM.[t_sub1, i] * A.[i, j]    
                                ) 0.0 [0..(A.RowCount-1)]
            let num = B.[t,j] * total
            //let alphaCol = alphaM.Column(i)
            //let d = Array.reduce (fun a b -> a+b) (alphaCol.ToArray())
            let d = alphaM.RowSums().Sum()
            num / d 

        // find q_t = argmax_{1 \le j \le N} (gamma_j), 1 \le t \le T
        // that is the argmax of state j at evidence time T where T is the index of V
        List.map (fun v -> 
                    let t = V.[v]
                    let t_sub1 = V.[v-1]
                    let row = List.map (fun j -> gammaFn j t t_sub1) [0..(A.ColumnCount-1)]
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
                    ) [1..((List.length V)-1)]

    (* 
    the training method makes use of the forward backward
    algorithm.

    input: the input model to start training
    trainSequences: the training sequences to present for learning, the last item in each sequence is considered to be the target state
    Note that training sequences need only be the set of complete sequences for each set of transitions.

    theta: the threshold to use until convergence
    maxEpochs: the maximum epochs to run if convergence is not met

    *)
    let train (input:InputModel) ( trainSequences:string list list) (theta:float) (maxEpochs:int) : Model = 
        let A = input.A
        let Bk = input.Bk
        let pi = input.pi
        let stateCount = List.length input.states 
        let evidenceCount = List.length input.evidence
        
        let matA = DenseMatrix.init A.RowCount A.ColumnCount (
                                        fun i j ->
                                            A.[i,j])
        
        let firstB = List.head Bk
        // the new B matrix will be multiplied against each sequence in Bk
        // so it is initialised to 1.
        let matB = DenseMatrix.init firstB.RowCount firstB.ColumnCount (
                            fun i j -> 1.0       
                            )

        let totalSequences = List.length trainSequences

        (*
        inner training function
        *)
        let rec innerTrain epoch error (matA, matB) : int * float * Matrix<float> * Matrix<float> =
            match epoch >= maxEpochs with
            | true -> (epoch, error, matA, matB)
            | false ->  
                // for each training sequence
                let (newA, newB) =
                    List.fold (fun (oldA, oldB) trainSeq ->
                            // select the subsequence of evidence vars
                            let example = trainSeq |> List.toSeq |> Seq.take ((List.length trainSeq) - 1) |> Seq.toList
                            let V = indices input.evidence example

                            //Console.WriteLine("Epoch {0} Indices {1}", epoch, V)
                           
                            // for each step in V we compute \hat{a_ij} and \hat{b_ij} first compute alpha and beta
                            let T = (List.length example) - 1
                            // the denominator is calculated for all B sequences

                            // the numerator is calculated only for the current e_t+1 in V for A 
                            // and only for the current e_t in V for B

                            // calculate the denominator

                            // firstly the normalisation factor P(O^k|model) this is the joint of the probability of the sequence in row B.[t, ]

                            let range = match T <= 1 with
                                        | true -> []
                                        | false -> [0..(T-1)]

                            // for each possible sequence in B
                            let (newA, newB) = 
                                List.fold(fun ((oldA, oldB):Matrix<float> * Matrix<float>) (B:Matrix<float>) ->

                                   (* 
                                    Console.WriteLine("oldA {0}{1}", Environment.NewLine, oldA)
                                    Console.WriteLine("oldB {0}{1}", Environment.NewLine, oldB)
                                    Console.WriteLine("B Matrix {0}{1}", Environment.NewLine, B)
                                    *)

                                    // since we are supplying a sequence B of matrice we will attempt to regularise

                                    // the B matrix by multiplying it with oldB
                                    let Bjoint = (DenseMatrix.init B.RowCount B.ColumnCount (
                                                                fun i j -> 
                                                                    B.[i,j] * oldB.[i,j]
                                                                    //pi.[j] * oldB.[i,j]
                                                                    //oldB.[i,j]
                                                                    )).NormalizeRows(1.0)
                                                                    //1.0 * oldB.[i,j])
                                    
                                    //Console.WriteLine("Bjoint {0}{1}", Environment.NewLine, Bjoint);

                                    // calculate the normalising factor from the joint distribution of V in B accross all states
                                    // for each index in V calculate the joint probability in all states 
                                    // multiply it with the joint probability of the other indices in V
                                    let P = List.fold(fun n t ->
                                                        List.map (fun i -> Bjoint.[t,i]) [0..(Bjoint.ColumnCount-1)]
                                                        |> List.reduce (fun a b -> a * b)
                                                        |> (fun p -> n + p)
                                                        ) 0.0 V

                                    let p = match P with
                                            | 0.0 -> 1.0
                                            | _ -> 1.0/P

                                    // calculate alpha and beta
                                    let alpha' = alpha (stateCount-1) pi oldA Bjoint stateCount evidenceCount V
                                    let beta' = beta (stateCount-1) pi oldA Bjoint stateCount evidenceCount V

                                    //Console.WriteLine("Alpha: {0}{1}", Environment.NewLine, alpha')
                                    //Console.WriteLine("Beta: {0}{1}", Environment.NewLine, beta')


                                    // fold over time t 2 times
                                    // first to calculate the denominator
                                    // this is the sum over all evidence vars e from 0 to T-1
                                    // in the matrix A (the rows of A 0..T-1
                                    let (denomA, denomB) = 
                                        List.fold (fun (d1, d2) t_index ->
                                                let e_t = V.[t_index]
                                                let e_tplus1 = V.[t_index+1]
                                                // in each state i
                                                List.fold (fun (a, b) i ->
                                                            let alpha_i = alpha'.[e_t, i]
                                                            let beta_i = beta'.[e_tplus1, i]
                                                            //let beta_i2 = beta'.[e_t, i]
                                                            let a' = a + (alpha_i * beta_i)
                                                            let b' = b + (alpha_i * beta_i)
                                                            (a', b')
                                                            ) (d1, d2) [0..(A.RowCount-1)]            
                                                
                                                ) (0.0, 0.0) [0..T-1]
                                        |> (fun (d1, d2) -> (p*d1, p*d2 ) )
                                        
                                    // now fold over time t to calculate the numerators
                                    // this is the probability in the sequence V (the range) from 0 to T-1
                                    let newA =
                                        List.fold(fun newA i ->
                                                        List.fold(fun (newA1:Matrix<float>) j ->
                                                                    // fold accross t in range (0..T-1)    
                                                                    let a = 
                                                                        match T <= 1 with
                                                                        | true -> 0.0
                                                                        | false ->
                                                                            List.fold(fun n t_index ->
                                                                                    let e_t = V.[t_index]
                                                                                    let e_tplus1 = V.[t_index+1]
                                                                                    let alpha_i = alpha'.[e_t, i]
                                                                                    let beta_j = beta'.[e_tplus1, j]
                                                                                    let a' = n + (alpha_i * matA.[i, j] * Bjoint.[e_tplus1, j] * beta_j)
                                                                                    a') 0.0 range
                                                                    newA1.[i, j] <- match denomA = 0.0 with
                                                                                    | true -> newA1.[i,j]
                                                                                    | false -> newA1.[i,j] + ( a / denomA )
                                                                    newA1
                                                                ) newA [0..(A.ColumnCount-1)]) oldA [0..(A.RowCount-1)]

                                    // fold over time T-1 calculate the normalising factor 
                                    // of e_t in time A
                                    
                                    // now from i to j we update a_ij
                                    let newB = 
                                        List.fold(fun (newB:Matrix<float>) (i:int) ->
                                            List.fold(fun (newB1:Matrix<float>) j ->
                                                        // fold accross t in range (0..T-1)    
                                                        let a =
                                                            List.fold(fun n t_index ->
                                                                        let e_t = V.[t_index]
                                                                        let alpha_j = alpha'.[e_t, j]
                                                                        let beta_i = beta'.[e_t, j]
                                                                        let a1 = n + alpha_j * beta_i
                                                                        a1) 0.0 [0..((List.length V) - 1)]
                                                        
                                                        newB1.[i,j] <- match denomB = 0.0 with
                                                                       | true -> newB1.[i,j]
                                                                       | false -> newB1.[i,j] + ( a / denomB )
                                                        newB1) newB [0..(B.ColumnCount-1)]
                                             ) Bjoint [0..(B.RowCount-1)]

                                    (newA.NormalizeRows(1.0), newB.NormalizeRows(1.0))) (oldA, oldB) Bk
           
                            (newA, newB)) (matA, matB) trainSequences
                // determine if deltas between oldA and oldB differ below threshold theta
                let normA = newA.NormalizeRows(1.0)
                let normB = newB.NormalizeRows(1.0)
                let deltaA = (normA-matA)
                let deltaB = (normB - matB)
                let absA = DenseMatrix.init deltaA.RowCount deltaA.ColumnCount (fun i j -> Math.Abs(deltaA.[i,j]))
                           |> (fun m -> m.RowSums().Sum())
                let absB = DenseMatrix.init deltaB.RowCount deltaB.ColumnCount (fun i j -> Math.Abs(deltaB.[i,j]))
                           |> (fun m -> m.RowSums().Sum())
                let max = match (absA > absB) with
                          | true -> absA
                          | _ -> absB
                // recurse if the changes are not small enough (not converging)
                match (max <= theta) with
                | true ->  (epoch, max, normA, normB)
                | _ -> innerTrain (epoch+1) max (normA, normB)
        // train sequences
        let (epoch, error, newA, newB) = innerTrain 1 0.0 (matA, matB)
        { pi = input.pi; 
          A = newA; 
          B = newB; 
          states = input.states; 
          evidence = input.evidence; 
          epoch = epoch;
          error = error; }

    (* 
    write the supplied model to file.
    *)
    let writeToFile (model:'a) (file:FileInfo) : bool =
        let binFormat = new BinaryFormatter()
        try 
            let outFile = new FileStream(file.FullName, FileMode.Create, FileAccess.Write, FileShare.None)
            binFormat.Serialize(outFile, model)
            outFile.Close()
            true
        with _ -> false

    (*
    read the supplied model from file
    *)
    let readFromFile (file:FileInfo) : Option<'a> =
        let binFormat = new BinaryFormatter()
        try
            let inFile = new FileStream(file.FullName, FileMode.Open, FileAccess.Read, FileShare.Read)
            let data = binFormat.Deserialize(inFile) :?> 'a
            inFile.Close()
            Some(data)
        with _ -> None
