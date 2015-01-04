

namespace au.id.cxd.HMMTestConsole

open au.id.cxd.HMM
open System
open System.IO
open au.id.cxd.HMM.HiddenMarkovModel
open au.id.cxd.HMM.DataTypes
open System.Collections.Generic
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra

module TestRain = 

    

    let test() =

        let data = 
            [["noumbrella"; "dry"];
             ["noumbrella"; "noumbrella"; "dry"];
             ["noumbrella"; "umbrella"; "rain"];
             ["umbrella"; "rain"];
             ["umbrella"; "umbrella"; "rain"];]

        let states = Reader.readStates data
        let evidenceVars = Reader.readEvidenceVars data

        let test = Estimation.countStateFreq data states
        let pi = Estimation.statePriors data states

        let T = Estimation.countTransitions data states
        let A = Estimation.stateTransitions data states

        // estimate sequence transition frequencies.
        let endSequences = Estimation.uniqueSequences data "rain"

        let test3 = List.head endSequences
        // [string list list] list - a list of subsequences
        let subSeqs = Estimation.allSubsequences data endSequences

        let Bk = [Estimation.priorEvidence data evidenceVars states]

        let endSequences2 = Estimation.uniqueSequences data "dry"
        let subSeq2 = Estimation.allSubsequences data endSequences2


        let inputModel = {
            pi = pi;
            A = A;
            Bk = Bk;
            states = states;
            evidence = evidenceVars;
            } 

        let model = HiddenMarkovModel.train inputModel endSequences 0.0005 10
        ()