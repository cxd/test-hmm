
#r "/Users/cd/Google Drive/Math/Markov Model/au.id.cxd.HMM/script/MathNet.Numerics.dll"
#r "/Users/cd/Google Drive/Math/Markov Model/au.id.cxd.HMM/script/MathNet.Numerics.FSharp.dll"
#load "Data/Reader.fs"
#load "Data/Estimation.fs"
#load "Model/DataTypes.fs"
#load "Model/HiddenMarkovModel.fs"
open au.id.cxd.HMM
open System
open System.IO
open au.id.cxd.HMM.HiddenMarkovModel
open System.Collections.Generic
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open au.id.cxd.HMM.DataTypes

// includes a placeholder "final" evidence var and an "end" state

let data = 
    [["noumbrella"; "dry"];
     ["noumbrella"; "noumbrella"; "dry"];
     ["noumbrella"; "noumbrella"; "noumbrella"; "dry"];
     ["noumbrella"; "umbrella"; "noumbrella"; "dry"];
     ["noumbrella"; "umbrella"; "noumbrella"; "noumbrella"; "dry"];


     ["noumbrella"; "umbrella"; "rain"];
     ["noumbrella"; "umbrella"; "noumbrella"; "rain"];
     ["noumbrella"; "umbrella"; "noumbrella"; "noumbrella"; "dry"];
     ["noumbrella"; "umbrella"; "noumbrella"; "noumbrella"; "umbrella"; "rain"];
     ["noumbrella"; "umbrella"; "noumbrella"; "umbrella"; "rain"];

     ["umbrella"; "rain"];
     ["umbrella"; "umbrella"; "rain"];
     ["umbrella"; "umbrella"; "umbrella"; "rain"];
    

     ["absent"; "storm"];
     ["umbrella"; "absent"; "storm"];
     ["absent"; "absent"; "absent"; "storm"];
    
     ["noumbrella"; "absent"; "storm"];
     ["noumbrella"; "absent"; "umbrella"; "rain"];
     ["noumbrella"; "absent"; "umbrella"; "absent"; "storm"];
     ["noumbrella"; "absent"; "absent"; "absent"; "umbrella"; "rain"];
     ["noumbrella"; "umbrella"; "absent"; "absent"; "storm"];
     ["noumbrella"; "absent"; "absent"; "absent"; "storm"];
     ["absent"; "absent"; "absent"; "absent"; "storm"];
 

     ]

let states = Reader.readStates data
let evidenceVars = Reader.readEvidenceVars data

let test = Estimation.countStateFreq data states
let pi = Estimation.statePriors data states

let T = Estimation.countTransitions data states
let A = Estimation.stateTransitions data states

let B1 = Estimation.evidenceTransition data evidenceVars states

let Bk = Estimation.avgPriorEvidences data evidenceVars states
//let Bk = Estimation.jointPriorEvidences subSeqs evidenceVars states


let inputModel = {
    pi = pi;
    A = A;
    Bk = [Bk];
    states = states;
    evidence = evidenceVars;
    } 


let model = HiddenMarkovModel.train inputModel data 0.0001 10


let predict = HiddenMarkovModel.predict model ["noumbrella"; "umbrella"; "umbrella"; "noumbrella"; "umbrella"; "umbrella";]

let testpredict = HiddenMarkovModel.viterbiPredict model ["noumbrella"; "umbrella"; "umbrella"; "noumbrella"; "umbrella"; "umbrella";]


let predict2 = HiddenMarkovModel.predict model ["noumbrella"; "umbrella"; "noumbrella"; "umbrella"; "umbrella"; "umbrella"; "umbrella";]


let predict3 = HiddenMarkovModel.predict model ["umbrella"; "absent"; "absent"]


let predict4 = HiddenMarkovModel.predict model ["noumbrella"; "noumbrella"; "absent";]

let predict5 = HiddenMarkovModel.predict model ["noumbrella"; "umbrella"; "noumbrella"; "umbrella"; "umbrella"; "umbrella"; "umbrella"; "absent";]


let predict6 = HiddenMarkovModel.predict model ["noumbrella"; "noumbrella"; "noumbrella"; "noumbrella"]

