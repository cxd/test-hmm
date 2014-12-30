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
open au.id.cxd.HMM.DataTypes
(*
This version of the module uses the approach outlined in AIMA section 15.2
*)
module HiddenMarkovModel = 

(*
The forward algorithm 

alphaScale is a normalising constant used to make probabilities sum to 1 for example 1 / sum(M)
T - is the state transition matrix for probability of transition from State i to State j = T_{ij} = P(S_j|S_i)
O - is the evidence/observation to state transition matrix. This is a set of matrices once for each state at time t.



*)
    let forward (alphaScale:float) (pi:float list) (T:Matrix<float>) (O:Matrix<float> list) =
        let prior = DenseVector.init (List.length pi) (fun i -> pi.[i])
        let rec innerForward (t:int) (alpha:float) (T:Matrix<float>) (O:Matrix<float> list) =
            match t = 0 with
            | true -> alpha * O.[t+1] * T.Transpose() * prior.ToColumnMatrix()
            | false -> 
                let cols = match t=1 with
                           | true -> [ innerForward t alpha T O ]
                           | false ->
                               List.map (fun t -> innerForward (t-1) alpha T O) [0..t]
                let mat = DenseMatrix.init T.RowCount (List.length cols) (fun i j -> 0.0)
                let temp = List.fold (fun i (col:Matrix<float>) -> 
                                            let col1 = col.Column(0) 
                                            mat.SetColumn(i, col1)
                                            i+1) 0 cols
                alpha * O.[t+1] * T.Transpose() * mat
        let t1 = (List.length O) - 2
        innerForward t1 alphaScale T O



    (* 
    the backward equation as defined in AIMA section 15.2

    this will only be able to function with at 3 or more evidence variables
    *)
    let backward (alphaScale:float) (pi:float list) (T:Matrix<float>) (O:Matrix<float> list) = 
        let prior = DenseVector.init (List.length pi) (fun i -> pi.[i])
        let len = (List.length O) - 3
        let rec inner (t:int) (T:Matrix<float>) (O:Matrix<float> list) =
            match (t = len) with
            | true -> T * O.[t+1] * prior.ToColumnMatrix()
            | false -> 
                let cols = 
                    match t=(len-1) with
                    | true -> [ inner (t+1) T O ] 
                    | false -> 
                        List.map (fun t1 -> inner t1 T O) [(t+1)..len]
                
                let mat = DenseMatrix.init T.RowCount (List.length cols) (fun i j -> 0.0)
                let temp = List.fold (fun i (col:Matrix<float>) -> 
                                            let col1 = col.Column(0) 
                                            mat.SetColumn(i, col1)
                                            i+1) 0 cols
                T * O.[t+1] * mat
        
        inner 0 T O

    ()


