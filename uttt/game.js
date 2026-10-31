export const LINES = [0b000000111,0b000111000,0b111000000,0b001001001,0b010010010,0b100100100,0b100010001,0b001010100];
export const POW = [1,3,9,27,81,243,729,2187,6561];
const won = (mask) => LINES.some(line => (mask & line) === line);

export class Game {
  constructor() {this.cells = new Uint8Array(81);this.boards = new Uint8Array(9);this.turn=1;this.target=9;this.winner=0;this.moves=[];this.last=-1;}
  localStatus(board) {
    let x=0,o=0,empty=0;
    for(let p=0;p<9;p++){const v=this.cells[board*9+p];if(v===1)x|=(1<<p);else if(v===2)o|=(1<<p);else empty++;}
    return won(x)?1:won(o)?2:empty?0:3;
  }
  globalStatus(){let x=0,o=0,closed=0;for(let b=0;b<9;b++){
    if(this.boards[b]===1)x|=1<<b; if(this.boards[b]===2)o|=1<<b;if(this.boards[b])closed++;
  }return won(x)?1:won(o)?2:closed===9?3:0;}
  isLegal(m){const b=(m/9)|0;return Number.isInteger(m)&&m>=0&&m<81&&!this.winner&&!this.boards[b]&&(this.target===9||this.target===b)&&!this.cells[m];}
  legal(){if(this.winner)return [];const out=[];const start=this.target===9?0:this.target,end=this.target===9?9:this.target+1;for(let b=start;b<end;b++)if(!this.boards[b])for(let p=0;p<9;p++)if(!this.cells[b*9+p])out.push(b*9+p);return out;}
  play(move){if(!this.isLegal(move))return false;const b=Math.floor(move/9),p=move%9;this.cells[move]=this.turn;const closed=this.localStatus(b);this.boards[b]=closed;if(closed)this.winner=this.globalStatus();this.target=this.boards[p]?9:p;this.moves.push(move);this.last=move;this.turn=3-this.turn;return true;}
  clone(){const g=new Game();g.cells.set(this.cells);g.boards.set(this.boards);g.target=this.target;g.turn=this.turn;g.winner=this.winner;g.moves=this.moves.slice();g.last=this.last;return g;}
  undo(n=1){const keep=this.moves.slice(0,Math.max(0,this.moves.length-n));this.reset();for(const m of keep)this.play(m);}
  reset(){this.cells.fill(0);this.boards.fill(0);this.target=9;this.turn=1;this.winner=0;this.moves=[];this.last=-1;}
}
export function cellName(m){if(m<0)return '—';const b=Math.floor(m/9),p=m%9;const r=Math.floor(b/3)*3+Math.floor(p/3),c=(b%3)*3+p%3;return `${String.fromCharCode(65+c)}${9-r}`;}
export function moveName(m){if(m<0)return '—';return `${String.fromCharCode(65+Math.floor(m/9))}${(m%9)+1}`;}
