#include <bits/stdc++.h>
using namespace std;
struct P { vector<pair<int,int>> c; string enc; };

vector<pair<int,int>> norm(vector<pair<int,int>> v){
  int minx=INT_MAX,miny=INT_MAX;
  for(auto [x,y]:v){minx=min(minx,x);miny=min(miny,y);} 
  for(auto &q:v){q.first-=minx;q.second-=miny;}
  sort(v.begin(),v.end(),[](auto a,auto b){return a.second!=b.second?a.second<b.second:a.first<b.first;});
  return v;
}
string encode(const vector<pair<int,int>>& v){
  string s; s.reserve(v.size()*5);
  for(size_t i=0;i<v.size();++i){ if(i)s.push_back(';'); s+=to_string(v[i].first); s.push_back(','); s+=to_string(v[i].second);} return s;
}
vector<pair<int,int>> transform(const vector<pair<int,int>>& c,int t){
  vector<pair<int,int>> v;v.reserve(c.size());
  for(auto [x,y]:c){int a,b;switch(t){
    case 0:a=x;b=y;break;case 1:a=-y;b=x;break;case 2:a=-x;b=-y;break;case 3:a=y;b=-x;break;
    case 4:a=-x;b=y;break;case 5:a=x;b=-y;break;case 6:a=y;b=x;break;default:a=-y;b=-x;break;
  }v.push_back({a,b});}return norm(move(v));
}
pair<string,vector<pair<int,int>>> canonical(const vector<pair<int,int>>& c){
  string best; vector<pair<int,int>> bestv; bool first=true;
  for(int t=0;t<8;t++){auto v=transform(c,t);auto e=encode(v);if(first||e<best){first=false;best=e;bestv=move(v);}}
  return {best,bestv};
}
bool hasHole(const vector<pair<int,int>>& c){
  int maxx=0,maxy=0;for(auto [x,y]:c){maxx=max(maxx,x);maxy=max(maxy,y);}int W=maxx+3,H=maxy+3;
  vector<char> occ(W*H,0),vis(W*H,0);for(auto [x,y]:c)occ[(y+1)*W+x+1]=1;
  deque<int> q;q.push_back(0);vis[0]=1;int dirs[4]={1,-1,0,0};
  while(!q.empty()){int id=q.front();q.pop_front();int x=id%W,y=id/W;int nx[4]={x+1,x-1,x,x},ny[4]={y,y,y+1,y-1};for(int d=0;d<4;d++){if(nx[d]<0||nx[d]>=W||ny[d]<0||ny[d]>=H)continue;int j=ny[d]*W+nx[d];if(!occ[j]&&!vis[j]){vis[j]=1;q.push_back(j);}}}
  for(int y=1;y<=maxy+1;y++)for(int x=1;x<=maxx+1;x++){int id=y*W+x;if(!occ[id]&&!vis[id])return true;}return false;
}
int perimeter(const vector<pair<int,int>>& c){unordered_set<long long>s;auto pack=[](int x,int y){return ((long long)x<<32)^(unsigned)y;};for(auto [x,y]:c)s.insert(pack(x,y));int p=0;for(auto [x,y]:c){p+=!s.count(pack(x+1,y));p+=!s.count(pack(x-1,y));p+=!s.count(pack(x,y+1));p+=!s.count(pack(x,y-1));}return p;}
int oriCount(const vector<pair<int,int>>& c){set<string> ss;for(int t=0;t<8;t++)ss.insert(encode(transform(c,t)));return (int)ss.size();}

int main(int argc,char**argv){int maxn=12;string outdir="data/catalog";if(argc>1)maxn=stoi(argv[1]);if(argc>2)outdir=argv[2];filesystem::create_directories(outdir);
  vector<P> cur={{{{0,0}},"0,0"}};
  vector<long long> expected={0,1,1,2,5,12,35,107,363,1248,4460,16094,58937};
  for(int n=1;n<=maxn;n++){
    if(n>1){unordered_map<string,vector<pair<int,int>>> m;m.reserve(cur.size()*5);
      for(auto &p:cur){unordered_set<long long> occ;auto pack=[](int x,int y){return ((long long)(x+64)<<32)^(unsigned)(y+64);};for(auto [x,y]:p.c)occ.insert(pack(x,y));
        for(auto [x,y]:p.c){const int dx[4]={1,-1,0,0},dy[4]={0,0,1,-1};for(int d=0;d<4;d++){int nx=x+dx[d],ny=y+dy[d];if(occ.count(pack(nx,ny)))continue;auto v=p.c;v.push_back({nx,ny});auto [e,cv]=canonical(v);if(!m.count(e))m.emplace(move(e),move(cv));}}
      }
      cur.clear();cur.reserve(m.size());for(auto &kv:m)cur.push_back({move(kv.second),kv.first});sort(cur.begin(),cur.end(),[](const P&a,const P&b){return a.enc<b.enc;});
    }
    vector<const P*> hf;hf.reserve(cur.size());for(auto &p:cur)if(!hasHole(p.c))hf.push_back(&p);
    cerr<<"n="<<n<<" free="<<cur.size()<<" holefree="<<hf.size();if(n<(int)expected.size())cerr<<" expected="<<expected[n]<<(hf.size()==expected[n]?" OK":" FAIL");cerr<<"\n";
    string path=outdir+"/free-holeless-n"+to_string(n)+".json";ofstream f(path);int pad=max(4,(int)to_string(hf.size()).size());
    f<<"{\"n\":"<<n<<",\"count\":"<<hf.size()<<",\"shapes\":[";
    for(size_t i=0;i<hf.size();i++){auto &c=hf[i]->c;int maxx=0,maxy=0;for(auto [x,y]:c){maxx=max(maxx,x);maxy=max(maxy,y);}if(i)f<<',';
      string num=to_string(i+1);num=string(max(0,pad-(int)num.size()),'0')+num;
      f<<"{\"id\":\"P"<<n<<"-"<<num<<"\",\"c\":\""<<hf[i]->enc<<"\",\"w\":"<<maxx+1<<",\"h\":"<<maxy+1<<",\"p\":"<<perimeter(c)<<",\"o\":"<<oriCount(c)<<"}";
    }
    f<<"]}\n";
  }
}
