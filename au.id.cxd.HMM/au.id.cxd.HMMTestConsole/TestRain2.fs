

namespace au.id.cxd.HMMTestConsole

open au.id.cxd.HMM
open System
open System.IO
open au.id.cxd.HMM.Multi
open au.id.cxd.HMM.Multi.MultiHiddenMarkovModel
open System.Collections.Generic
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra

module TestRain2 = 

    let test() = 
        // includes a placeholder "final" evidence var and an "end" state

        let data = 
            [["noumbrella"; "dry"];
             ["noumbrella"; "noumbrella"; "dry"];
             ["noumbrella"; "noumbrella"; "noumbrella"; "dry"];
             ["noumbrella"; "umbrella"; "noumbrella"; "dry"];
             ["noumbrella"; "umbrella"; "noumbrella"; "noumbrella"; "dry"];


             ["noumbrella"; "umbrella"; "rain"];
             ["noumbrella"; "umbrella"; "noumbrella"; "rain"];
             ["noumbrella"; "umbrella"; "noumbrella"; "noumbrella"; "rain"];
             ["noumbrella"; "umbrella"; "noumbrella"; "noumbrella"; "umbrella"; "rain"];
             ["noumbrella"; "umbrella"; "noumbrella"; "umbrella"; "rain"];

             ["umbrella"; "rain"];
             ["umbrella"; "umbrella"; "rain"];
             ["umbrella"; "umbrella"; "umbrella"; "rain"];
            

             ["absent"; "storm"];
             ["umbrella"; "absent"; "storm"];
             ["absent"; "absent"; "absent"; "storm"];
            
             ["noumbrella"; "absent"; "storm"];
             ["noumbrella"; "absent"; "noumbrella"; "rain"];
             ["noumbrella"; "absent"; "noumbrella"; "absent"; "storm"];
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

        let Bk = Estimation.priorEvidence data evidenceVars states


        let inputModel = {
            pi = pi;
            A = A;
            Bk = [Bk];
            states = states;
            evidence = evidenceVars;
            epoch = 10;
            error = 0.0001
            } 


        let model = MultiHiddenMarkovModel.train inputModel data


        let predict = MultiHiddenMarkovModel.predict model ["noumbrella"; "umbrella"; "umbrella"; "noumbrella"; "umbrella"; "umbrella";]

        let predict2 = MultiHiddenMarkovModel.predict model ["noumbrella"; "umbrella"; "noumbrella"; "umbrella"; "umbrella"; "umbrella"; "umbrella";]


        let predict3 = MultiHiddenMarkovModel.predict model ["umbrella"; "absent"; "absent"]


        let predict4 = MultiHiddenMarkovModel.predict model ["noumbrella"; "noumbrella"; "absent";]

        let predict5 = MultiHiddenMarkovModel.predict model ["noumbrella"; "umbrella"; "noumbrella"; "umbrella"; "umbrella"; "umbrella"; "umbrella"; "absent";]


        let predict6 = MultiHiddenMarkovModel.predict model ["noumbrella"; "noumbrella"; "noumbrella"; "noumbrella"]

        ()
