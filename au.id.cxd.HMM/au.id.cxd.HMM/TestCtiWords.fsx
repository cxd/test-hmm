#r "D:/ntdev/research/test-hmm/au.id.cxd.HMM/script/MathNet.Numerics.dll"
#r "D:/ntdev/research/test-hmm/au.id.cxd.HMM/script/MathNet.Numerics.FSharp.dll"

#load "Data/Reader.fs"
#load "Data/Estimation.fs"
#load "Model/HiddenMarkovModel.fs"

open au.id.cxd.HMM
open System
open System.IO
open au.id.cxd.HMM.HiddenMarkovModel
open System.Collections.Generic
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra

(* train given supplied filename and end state *)
let train filename endState =
    let fInfo = new FileInfo(filename)
    let data = Reader.readSequences fInfo
    let states = Reader.readStates data
    let evidenceVars = Reader.readEvidenceVars data

    let pi = Estimation.statePriors data states
    let A = Estimation.stateTransitions data states
    // TODO: estimate sequence transition frequencies.
    let endSequences = Estimation.uniqueSequences data "endState"
    let subSeqs = Estimation.allSubsequences data endSequences
    let Bk = [Estimation.avgPriorEvidences subSeqs evidenceVars states]
    let inputModel = {
    pi = pi;
    A = A;
    Bk = Bk;
    states = states;
    evidence = evidenceVars;
    } 

    // let model = HiddenMarkovModel.train inputModel

    let stateCount = List.length states
    let evidenceCount = List.length evidenceVars
    HiddenMarkovModel.train inputModel data 0.0001 500

let maxEstimate (models:Model list) (evidence:string list) =
    let results = List.map (fun model -> 
                                let predict = HiddenMarkovModel.predict model evidence
                                let sum = List.map (fun p -> 
                                                        match p.state with
                                                        | "None" -> 0.0
                                                        | _ -> Math.Log(p.prob)) predict
                                          |> List.reduce (fun a b -> a + b)
                                (sum, predict)) models
    List.maxBy (fun (sum, predict) -> sum) results



let baseDir = "D:/ntdev/research/test-hmm/au.id.cxd.HMM/au.id.cxd.HMMTestConsole/data/cti/"
// each of these represent a file and a end state
let files = [("oncall.txt", "OnCall");
             ("started.txt", "Started");
             ("paused.txt", "Paused");
             ("held.txt", "OnHold");
             ("released.txt", "Ended");
             ("consult.txt", "Consult");]
let models = List.map (fun (file, state) -> train (baseDir + file) state) files
let test1 = ["Ringing(inbound)";"UserEvent(Start)";"UserEvent(Stop)";"OffHook";"Established";"Dialing(Consult)"];
maxEstimate models test1