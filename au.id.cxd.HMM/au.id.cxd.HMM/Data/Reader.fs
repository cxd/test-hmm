namespace au.id.cxd.HMM

open System
open System.IO

module Reader =

    (*
        read lines from supplied file info
    *)
    let readLines (file:FileInfo): string list = 
        let rec innerReadLines (accum:string list) (reader:StreamReader) =
               let line = 
                   try 
                    let d = reader.ReadLine()
                    match (String.IsNullOrEmpty(d)) with
                    | false -> Some(d)
                    | _ -> None
                   with 
                   | _ -> None
               match line with
               | Some(data:string) -> innerReadLines (data::accum) reader
               | None -> accum
        let reader = new StreamReader(file.FullName)
        let lines = innerReadLines [] reader
        reader.Close() |> ignore
        lines

    (*
    Read the list of sequences from the supplied file
    *)
    let readSequences (file:FileInfo): string list list = 
        let lines = readLines file
        List.fold (
            fun (accum:string list list) (line:string) -> 
                match line.Trim().StartsWith("#") with
                | true -> accum
                | _ -> ( (line.Trim().Split([|','|]) |> Array.toList) |> List.map(fun item -> item.Trim()) ) :: accum) [] lines
 

    (* extract the unique items from the list *)
    let unique (items:string list) =
         List.fold(fun accum token ->
                    let ismatch (term:string) = List.exists (fun (other:string) -> term.ToLower().Equals(other.ToLower()))
                    match (ismatch token accum) with
                    | true -> accum
                    | false -> token :: accum) [] items   
        
            

(* 
determine the unique state labels
The last item of each sequence is considered to be the state label.
*)
    let readStates (entries:string list list) =
        List.fold(fun accum items -> 
                        (List.rev items |> List.head) :: accum
                        ) [] entries
        |> unique
            


(* 
determine the unique evidence variables
all items except the last items are considered evidence variables 
*)
    let readEvidenceVars (entries:string list list) =
        List.fold(fun accum items ->
                    List.concat (seq [ (List.rev items |> List.tail); accum ]) ) [] entries
        |> unique
        

