// v11 shared, rule-derived endgame phase detector. Neither time control nor a
// fixed ply/empty-cell threshold determines when an endgame starts.
export function endgameProfile(wasm, complexity=null) {
 const signal=wasm.apex_endgame_signal?.()??0;
 const viableX=wasm.apex_macro_viable_lines?.(1)??8;
 const viableO=wasm.apex_macro_viable_lines?.(2)??8;
 const activeBoards=wasm.apex_active_boards?.()??9;
 const structuralDraw=!!wasm.apex_structural_draw?.();
 const rootLegal=wasm.apex_root_legal_count?.()??9;
 const effectivelyNarrow=complexity?.effectiveBranch<=1.7||
  (complexity?.forced??0)>=.60 || (complexity?.hitRate??0)>=.42;
 const projectedCheap=Number.isFinite(complexity?.projectedNodes)&&complexity.projectedNodes<600000;
 // Complexity and macro-line topology can trigger endgame search early,
 // including when dozens of legal squares remain in still-open boards.
 const urgency=Math.min(1,signal/1000+
  (effectivelyNarrow?.12:0)+(projectedCheap?.12:0));
 const focused=structuralDraw||urgency>=.38||
  (urgency>=.27&&(effectivelyNarrow||projectedCheap));
 const proofPriority=structuralDraw||urgency>=.33||
  (urgency>=.19&&(effectivelyNarrow||projectedCheap));
 // Only an unambiguous structural endgame may take over the full budget.
 // Topology matters: a high tactical threat count on nine still-open boards
 // must not suppress normal search in the midgame.
 const definite=structuralDraw||
  (activeBoards<=6 && urgency>=.60 && viableX+viableO<=8)||
  (activeBoards<=7 && urgency>=.77 && viableX+viableO<=7)||
  (urgency>=.94 && activeBoards<=7);
 const borderline=!definite&&focused;
 const pvsShare=definite?Math.min(.98,Math.max(.88,.63+.50*urgency)):
  borderline?Math.min(.62,Math.max(.36,.22+.45*urgency)):
  Math.min(.27,Math.max(.06,.10*urgency));
 return {signal,urgency,focused,proofPriority,definite,borderline,pvsShare,rootLegal,
  activeBoards,viableX,viableO,structuralDraw,effectivelyNarrow,projectedCheap};
}
