namespace au.id.cxd.HMM

open System
open System
open System.IO
open System.Text
open System.Collections.Generic
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open System.Runtime.Serialization
open System.Runtime.Serialization.Formatters.Binary

module DataTypes =

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


    let show (predict:Prediction) =
            let sb = new StringBuilder()
            sb.AppendFormat("[ prob: {0},", predict.prob)
                .AppendLine()
               .AppendFormat("state: {0},", predict.state)
                .AppendLine()
               .AppendFormat("evidence: {0},", predict.evidence)
                .AppendLine()
               .AppendFormat("t: {0},", predict.t)
                .AppendLine()
               .AppendFormat("success: {0}", predict.success)
                .AppendLine()
               .AppendFormat("]")
                .AppendLine()
               .ToString()
