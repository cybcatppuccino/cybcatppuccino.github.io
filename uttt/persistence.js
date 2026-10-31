import {Game} from './game.js';
// Replay the complete move history instead of trusting a saved board bitmap.
// This restores last move, target board, closed mini boards, undo, and side to move.
export function replayMoves(moves){
 if(!Array.isArray(moves)||moves.length>81)return null;
 const game=new Game();
 for(const m of moves)if(!Number.isInteger(m)||!game.play(m))return null;
 return game;
}
