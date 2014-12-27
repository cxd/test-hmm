//#r "D:/ntdev/research/test-hmm/au.id.cxd.HMM/script/MathNet.Numerics.dll"
//#r "D:/ntdev/research/test-hmm/au.id.cxd.HMM/script/MathNet.Numerics.FSharp.dll"

#r "/Users/cd/Google Drive/Math/Markov Model/au.id.cxd.HMM/script/MathNet.Numerics.dll"
#r "/Users/cd/Google Drive/Math/Markov Model/au.id.cxd.HMM/script/MathNet.Numerics.FSharp.dll"

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
    let endSequences = Estimation.uniqueSequences data endState
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
                                Console.WriteLine("Predict{0}{1}", Environment.NewLine, List.map HiddenMarkovModel.show predict)
                                let sum = List.map (fun p -> 
                                                        match p.state with
                                                        | "None" -> 0.0
                                                        | _ -> Math.Abs(Math.Log(p.prob))) predict
                                          |> List.reduce (fun a b -> a + b)
                                Console.WriteLine("SUM: {0}", sum)
                                (sum, predict)) models
    List.maxBy (fun (sum, predict) -> sum) results
    //results


//let baseDir = "D:/ntdev/research/test-hmm/au.id.cxd.HMM/au.id.cxd.HMMTestConsole/data/cti/"
let baseDir = "/Users/cd/Google Drive/Math/Markov Model/au.id.cxd.HMM/au.id.cxd.HMMTestConsole/data/cti/"
// each of these represent a file and a end state
let files = [("oncall.txt", "OnCall");
             ("started.txt", "Started");
             ("consult.txt", "Consult");
             ("paused.txt", "Paused");
             ("held.txt", "OnHold");
             ("released.txt", "Ended");
             ]
            
let models = List.map (fun (file, state) -> train (baseDir + file) state) files
let test1 = ["Ringing(inbound)";"UserEvent(Start)";"UserEvent(Stop)";"OffHook";"Established";"Dialing(Consult)"];
maxEstimate models test1

let test2 = ["Ringing(inbound)";"UserEvent(Start)";"UserEvent(Stop)";"OffHook";"Established";"Held";];
maxEstimate models test2

let test3 = ["Ringing(inbound)";"UserEvent(Start)";];
maxEstimate models test3