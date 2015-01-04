namespace au.id.cxd.HMM.Log

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


    (* retrieve the indices for supplied index *)
    let indices (V:string list) (index:string list) =
        List.map (fun (item:string) ->
                    List.findIndex (fun (key:string) -> key.Equals(item, StringComparison.OrdinalIgnoreCase)) index
                    ) V


    (* logarithm calculations
    based on the article by Tobias Mann 2006.
    *)

    let log_zero = 0.0

    (*
    logarithm extended to handle log zero
    *)
    let ln x = 
        match x = 0.0 with
        | true -> log_zero
        | false -> 
            match x > 0.0 with
            | true -> Math.Log(x)
            | false -> 
                // assume that x is already in log space.
                x

    (*
    exponent extended to handle log_zero
    *)
    let exp x =
        match x = log_zero with
        | true -> 0.0
        | false -> 
            let e = Math.Exp(x)
            if Double.IsNaN(e) then 0.0
            else if Double.IsInfinity(e) then 1.0
            else e

    (*
    calculate the logarithn sum of x and y

    the sum formula is derived as

    $$
    ln(x) + (ln(1 + e^{ln(y) - ln(x)})) = ln(x) + ln(1 + e^{ln(y/x)})
    = ln(x) + ln(1 + \frac{y}{x})
    = ln(x) + ln(\frac{x + y}{x})
    = ln(x) + ln(x + y) - ln(x)
    = ln(x + y)

    $$
    *)
    let lnsum x y =
        match (x=0.0 || y=0.0) with
        | true ->
            match x = 0.0 with
            | true -> y
            | false -> x
        | false ->
            match x > y with
            | true -> x + ln(1.0+exp(y-x))
            | false -> y + ln(1.0 + exp(x-y))

    (*
    Log product (ln x + ln y)
    *)
    let lnprod x y =
        x + y

    (* forward equation
   
    T = time T
    pi = P(x_i)
    a_ij = P(x_j | x_i)
   b_ij = P(x_j | e_i)

   T represents i current state we need to predict j next state

   V - indices of evidence variables observed up to time T

   the values are computed in log space (to convert back to probabilities use exp)
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

            this is computed in log space

            *)
            let inner (vk:int) (t:int) (accum:Matrix<float>):Matrix<float> =
                //Console.WriteLine("Accum: {0}", accum)

                let alpha2 = cache (t-1) (fun () -> alphaInner (t - 1) pi A B V accum)

                let totals = DenseMatrix.init accum.RowCount accum.ColumnCount (
                                    fun i j ->
                                        lnprod alpha2.[i, t - 1] (ln A.[t-1, j]))
                let total = List.map (fun j -> 
                                        let col = totals.Column(j).ToArray()
                                        let sum = Array.reduce lnsum col
                                        lnprod sum (ln B.[vk, t])) [0..(accum.ColumnCount-1)]
                accum.SetRow(vk, List.toArray total)
                accum

            match T = 0 with
            | true -> accum
            | _ -> List.fold (fun (accum:Matrix<float>) (k:int) -> inner k T accum) accum V

        let accum = DenseMatrix.init evidenceCount stateCount (
                        fun t j ->
                            let vlen = List.length V
                            match t < vlen with
                            | true -> 
                                let idx = V.[t]
                                lnprod (ln pi.[j]) (ln B.[idx, j])
                            | false -> ln pi.[j] // (Convert.ToDouble(evidenceCount))
                    )
        
        Console.WriteLine("Start Dim: [{0}, {1}]", accum.RowCount, accum.ColumnCount)
        Console.WriteLine("Start: {0}{1}", Environment.NewLine, accum)        
            
        let alphaResult = alphaInner T pi A B V accum
        alphaResult
        

    (*
    this is the time reversed algorithm of alpha
    moving from t = 1 to T

    T = time T
    pi = P(x_i)
    a_ij = P(x_j | x_i)
   b_ij = P(x_j | e_i)

   T represents i current state we need to predict j next state

   V - indices of evidence variables observed up to time T

   this is performed in log space
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

                // \beta_i(t) = \sum_{j=1}^C \beta_j(t+1)a_{ij}b_{jk}v(t+1)
                let totals = DenseMatrix.init accum.RowCount accum.ColumnCount (
                                    fun i j ->
                                        lnprod beta2.[i, t + 1] (lnprod (ln A.[t, j]) (ln B.[vk, t + 1]) ) ) 
                let total = List.map (fun j -> 
                                        let col = totals.Column(j).ToArray()
                                        let sum = Array.reduce lnsum col
                                        sum) [0..(accum.ColumnCount-1)]
                accum.SetRow(vk, List.toArray total)
                accum
            
            match t >= (T-1) with
            | true -> accum
            | false -> List.fold (fun (accum:Matrix<float>) (k:int) -> inner k (t+1) accum) accum V

        // initialising beta matrix with priors
        let accum = DenseMatrix.init evidenceCount stateCount (
                        fun t j ->
                            let vlen = List.length V
                            match t < vlen with
                            | true -> let idx = V.[t]
                                      lnprod (ln pi.[j]) (ln B.[idx, j])
                            | false -> lnprod (ln pi.[j]) (ln (1.0/Convert.ToDouble(evidenceCount))) 
                    )

        Console.WriteLine("Start Dim: [{0}, {1}]", accum.RowCount, accum.ColumnCount)
        Console.WriteLine("Start: {0}{1}", Environment.NewLine, accum) 

        //let test = DenseMatrix.init (List.length V) A.ColumnCount ( fun i j -> 1.0 )
        let betaResult = betaInner T 0 pi A B V accum
        betaResult


            (*
Compute the gamma matrix given the alpha and beta matrix
normalise by calculating the normalisation factor.
Calculations are performed in log space
    *)

    let gammaFn (alphaM:Matrix<float>) (betaM:Matrix<float>) :Matrix<float> =
        let gamma = DenseMatrix.init alphaM.RowCount alphaM.ColumnCount (fun i j -> 0.0) 
        let (n, G) = 
            List.fold (fun (n, G) e_i ->
                    List.fold(fun (n, G:Matrix<float>) s_j ->
                                G.[e_i, s_j] <- lnprod alphaM.[e_i, s_j] betaM.[e_i, s_j]
                                let n1 = lnsum n G.[e_i, s_j]
                                (n1, G)) (n, G) [0..(alphaM.ColumnCount - 1)]
                    ) (0.0, gamma) [0..(alphaM.RowCount-1)]
        DenseMatrix.init G.RowCount G.ColumnCount (fun i j -> lnprod G.[i, j] n)

     (*
        compute epsilon function for each evidence variable at time t
        returns a list of state transition matrices of size A_i, A_j for each observation in O_t
        so 
     *)
    let epsilonFn (A:Matrix<float>) (B:Matrix<float>) (alphaM:Matrix<float>) (betaM:Matrix<float>) : Matrix<float> list =
         let epsilonT n E t =
            List.fold(fun (n, E) s_i ->
                     List.fold(fun (n, E:Matrix<float>) s_j ->
                                E.[s_i, s_j] <- lnprod alphaM.[t, s_i] (lnprod (ln A.[s_i, s_j]) (lnprod (ln B.[t+1, s_j]) betaM.[t+1, s_j]))
                                let n1 = lnsum n E.[s_i, s_j]
                                (n1, E)) (n, E) [0..A.ColumnCount - 1]
                                ) (n, E) [0..A.RowCount-1]
         List.fold (fun (n, accum) t ->
                    let E = DenseMatrix.init A.RowCount B.RowCount (fun i j -> 1.0)
                    let (n1, E1) = epsilonT n E t
                    (n1, E::accum)
                    ) (0.0, []) [0..(alphaM.RowCount - 2)]
         |> (fun (n, accum) ->
               let accum1 = List.rev accum
               List.map (fun (A1:Matrix<float>) -> 
                            DenseMatrix.init A1.RowCount A1.ColumnCount (
                                    fun i j -> lnprod (ln A1.[i, j]) -1.0*n) ) accum1)

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
        let V = indices evidenceSequence model.evidence
        let T = (List.length model.states) - 1
        let stateCount = List.length model.states
        let evidenceCount = List.length model.evidence
        let alphaM = alpha T pi A B stateCount evidenceCount V
        let betaM = beta T pi A B stateCount evidenceCount V

        let gammaM = gammaFn alphaM betaM

        // find q_t = argmax_{1 \le j \le N} (gamma_j), 1 \le t \le T
        // that is the argmax of state j at evidence time T where T is the index of V
        List.map (fun v -> 
                    let t = V.[v]
                    let row = gammaM.Row(t).ToArray() |> Array.toList
                    Console.WriteLine("Time: {0} Row: {1}", t, List.fold(fun (sb:StringBuilder) i -> sb.Append(i.ToString()).Append(" ")) (new StringBuilder()) row)
                    let max = Double.MinValue
                    let (max, index, searchIdx, flag) = 
                        List.fold(
                            fun (max, index, searchIdx, flag) p ->
                                let prob = exp(p)
                                match (prob > max) with
                                | true -> (prob, searchIdx, searchIdx+1, true)
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

        // use the joint distribution of the B matrices
        let matB1 = List.fold(fun m (b:Matrix<float>) ->
                                DenseMatrix.init m.RowCount m.ColumnCount (
                                    fun i j -> 
                                        lnprod m.[i,j] (ln b.[i,j]) )
                                ) matB Bk

        let totalSequences = List.length trainSequences

        (*
        inner training function
        *)
        let rec innerTrain epoch error (pPi, matA, matB) : int * float * float list * Matrix<float> * Matrix<float> =
            match epoch >= maxEpochs with
            | true -> (epoch, error, pPi, matA, matB)
            | false ->  
                // for each training sequence

                let (newPi, newA, newB) =
                    List.fold (
                        fun (oldPi, oldA, oldB) trainSeq ->
                            let example = trainSeq |> List.toSeq |> Seq.take ((List.length trainSeq) - 1) |> Seq.toList
                            let V = indices example input.evidence

                            // for each step in V we compute \hat{a_ij} and \hat{b_ij} first compute alpha and beta
                            let T = (List.length example) - 1
                            

                            let alpha1 = alpha (stateCount-1) pi oldA oldB stateCount evidenceCount V
                            let beta1 = beta (stateCount-1) pi oldA oldB stateCount evidenceCount V
                            // gamma function
                            let gammaM = gammaFn alpha1 beta1
                            // epsilon for all states
                            let epsilonM = epsilonFn oldA oldB alpha1 beta1

                            let newPi = oldPi

                            // compute the estimated probability of transitioning from S_i to S_j
                            let newA = DenseMatrix.init oldA.RowCount oldA.ColumnCount (
                                            fun i j ->
                                                let (n, d) = 
                                                     List.fold( fun (n, d) tidx ->
                                                                    let t = V.[tidx]
                                                                    let E = epsilonM.[t]

                                                                    let n1 = n + E.[i, j]
                                                                    let d1 = d + gammaM.[t, i]
                                                                    (n1, d1)) (0.0, 0.0) [0..(T-1)]
                                                exp(lnprod n -1.0*d))
                                                

                            let newB = DenseMatrix.init oldB.RowCount oldB.ColumnCount (
                                        fun k j ->
                                            // calculate the numerator and denominator.
                                            // the numerator is only for the current evidence variable t
                                            // the denomiator is for all evidence variables
                                            let (n, d) =
                                                List.fold(
                                                    fun (n,d) tidx ->
                                                        let t = V.[tidx]
                                                        let n1 = 
                                                            match t = k with
                                                            | true -> n + gammaM.[t, j]
                                                            | false -> n
                                                        let d1 = d + gammaM.[t, j]
                                                        (n1,d1)) (0.0, 0.0) [0..T]
                                            exp (lnprod n -1.0*d)
                                        )
                            
                            (newPi, newA, newB)) (pi, matA, matB) (trainSequences |> List.filter (fun subseq -> List.length subseq > 2))
                // determine if deltas between oldA and oldB differ below threshold theta
                let deltaA = (newA - matA)
                let deltaB = (newB - matB)
                let absA = DenseMatrix.init deltaA.RowCount deltaA.ColumnCount (fun i j -> Math.Abs(deltaA.[i,j]))
                           |> (fun m -> m.RowSums().Sum())
                let absB = DenseMatrix.init deltaB.RowCount deltaB.ColumnCount (fun i j -> Math.Abs(deltaB.[i,j]))
                           |> (fun m -> m.RowSums().Sum())
                let max = match (absA > absB) with
                          | true -> absA
                          | _ -> absB
                
                Console.WriteLine("Epoch: {0}", epoch)
                Console.WriteLine("A {0}{1}", Environment.NewLine, newA)
                Console.WriteLine("B {0}{1}", Environment.NewLine, newB)

                // recurse if the changes are not small enough (not converging)
                match (max <= theta) with
                | true ->  (epoch, max, pi, newA, newB)
                | _ -> innerTrain (epoch+1) max (pi, newA, newB)
        // train sequences
        let (epochs, error, pi, newA, newB) = innerTrain 1 0.0 (pi, matA, matB1)
        { pi = input.pi; 
          A = newA; 
          B = newB; 
          states = input.states; 
          evidence = input.evidence;
          epoch = epochs;
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
