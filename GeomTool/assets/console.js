/* GeomTool v8 command catalog, fuzzy offline search and help (no network). */
(function(root,factory){const api=factory();root.GeomConsole=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){'use strict';
const sections=[
 ['常用命令与对象',[
  ['help [关键词]','弹出独立命令手册；内含分类折叠、实时智能搜索和命令示例','help 旋转','帮助 手册 搜索'],
  ['A / select A','输入唯一对象名称直接选中（包括隐藏、退化及系统对象），并打开属性','p_0','选中 对象 属性'],
  ['list [all|hidden|visible|selected]','列出对象与可见状态；hidden 不会将对象从列表剔除','list hidden','对象 列表'],
  ['hide A / show A / toggle A','独立改变对象显示；坐标轴总开关关闭时系统对象仍不绘制','hide l_Re','可见 隐藏 显示'],
  ['rename A 新名 / delete A','重命名或删除对象，派生对象依赖关系由内核维护；系统对象不允许改名/删除','rename A P','名称 删除'],
  ['style A dashed on|off / color A #ff8800','编辑对象虚线或十六进制颜色；style A visible on/off 亦可','style l1 dashed on','样式 色彩'],
  ['move A x y / drag A x y','按世界坐标移动自由点或活动点，可输入表达式','move A 2pi/3 1','拖拽 定位'],
  ['undo / redo / clear','撤销、重做、清空（有撤销记录）','undo','历史 画板'],
 ]],
 ['几何构造与工具',[
  ['point A x y / real R x','创建平面自由点或实轴点；支持 π、sqrt、分数','point A -2 sqrt(3)','构造 坐标'],
  ['line l A B / segment s A B / ray r A B','创建直线、线段、射线；其他构造可使用内核对应英文操作名','line l A B','作图'],
  ['circle c O P / circler c O 2','圆心过点或定半径圆；circler 可接受数学表达式','circler c O 2pi/3','圆 半径'],
  ['rotate R P O π/2 rad','旋转构造：最后可选 deg（默认）或 rad；弧度支持 pi、π、2pi/3','rotate R P O 2π/3 rad','旋转 弧度'],
  ['homothety H P O 3/2','位似构造：输入倍数表达式；也支持 -sqrt(2)','homothety H P O -sqrt(2)','放缩 倍数'],
  ['tool rotate / group transform','切换左侧当前工具或工具组；工具名可使用内核方法名','tool homothety','工具栏'],
  ['param 值 / unit deg|rad','设置左侧当前构造参数；旋转支持角度/弧度切换','unit rad','输入控件 π'],
  ['step A / finish / cancel','对当前鼠标构造工具添加已存在对象作为输入；完成多边形或取消未完成步骤','tool segment','鼠标 输入'],
  ['click x y / smart x y','按世界坐标执行当前工具单击；智能点自动吸附；双载体求交点','smart 0 1','点击 智能点'],
  ['methods / guide rotate','列出全部内核构造语法，或查看指定工具的操作指引','guide rotate','所有构造 模板'],
 ]],
 ['工作区、外观和视图',[
  ['settings / settings show','打开工作区设置弹窗；settings close 关闭','settings show','工作区 首选项'],
  ['settings grid on|off / labels on|off','显示/隐藏网格与点标签；亦可使用 setting 单数','settings grid on','网格 标签'],
  ['settings axes on|off / complex on|off','切换坐标轴及复平面；关闭坐标轴会关掉复平面','settings axes on','坐标轴 基准点'],
  ['settings snap on|off / wheel on|off','配置辅助作图吸附及滚轮缩放','settings snap off','交互 设置'],
  ['settings precision 0..15 / palette blue|graphite|emerald','设置小数显示精度与绘图颜色','settings precision 8','精度 调色板'],
  ['settings theme light|dark / theme toggle','切换日间/夜间模式','theme dark','深色 外观'],
  ['settings leftWidth 300 / rightWidth 400','调整侧栏宽度；还可以设置 leftCollapsed、rightCollapsed','settings leftWidth 310','栏位 布局'],
  ['zoom in|out|125% / fit / pan dx dy','缩放、适配全部对象、按像素平移画布；缩放亦可输入倍率','zoom 150%','缩放 视图 平移'],
  ['center x y / resetview','世界坐标点居中；或恢复 100% 视图','center 0 0','居中 复位'],
  ['panel objects|details|pretty|relations|complex|proof|library|left|right','展开目标工作台折叠区或侧栏','panel relations','工作台 折叠'],
 ]],
 ['Pretty 画面美化',[
  ['beautify [力度] / pretty run [力度]','以指定力度（5–100）执行 Pretty 自动布局并适配视图','beautify 65','美化 梯度'],
  ['pretty strength 5..100','设置 Pretty 优化力度滑杆','pretty strength 80','优化 力度'],
  ['pretty progress 0..100','沿计算后的优化轨迹移动进度滑杆','pretty progress 50','回看 进度'],
  ['pretty status','读取 Pretty 最近一次计算状态','pretty status','状态'],
 ]],
 ['关系发掘系统',[
  ['relations scan / relations clear','扫描已勾选类别，或清除当前关系结果','relations scan','雷达 扫描'],
  ['relations kinds [all|none|类别 on|off]','列出关系类别或更改勾选；类别可使用检测 ID 或中文名','relations kinds all','类别 检测'],
  ['relations hidden on|off / perturb on|off','搜索隐藏对象及微扰复验开关','relations hidden on','隐藏对象 复验'],
  ['relations quadratic product|expression [0..200]','二次量默认关闭：设置乘积和平方和差的枚举操作数上限；扫描后显示截断数','relations quadratic expression 50','二次量 性能 枚举 上限'],
  ['relations search 关键词 / sort score|kind','检索关系结果并选择排序方式','relations search A B','检索 排序'],
  ['relations select 序号 / more / list','选中高亮一个当前结果（从 1 开始）、加载更多或列出结果','relations select 1','结果 高亮'],
 ]],
 ['DDAR 形式推理（v8）',[
  ['ddar run','执行原生 AG2 DDAR 几何闭包，不枚举数值候选','ddar run','DDAR 推理 形式证明'],
  ['ddar prove','证明欲证列表当前选中的命题','ddar prove','待证目标'],
  
  
  ['ddar stats','显示 DD 轮数、内核日志条数和运行时间','ddar stats','结论 统计'],
  ['ddar export','输出本次 AG2 输入与已选命题的 JSON','ddar export','证明 数据'],
  ['panel ddar','展开右侧「结论论证」区域','panel ddar','面板 推理'],
 ]],
 ['几何资料库',[
  ['library search 关键词 / filter 分类','搜索离线资料库或按目录筛选','library search Euler','几何资料库'],
  ['library list / select 序号|ID / more','列出匹配题库、打开条目、显示下一页','library select 1','资料库 选择'],
  ['library load ID / library load selected','按记录 ID 或当前条目加载并自动 Pretty（覆盖当前画板前确认）','library load selected','题库 载入'],
 ]],
 ['导入、导出和文档',[
  ['export / export file / export json','下载 .geom.json 文件，或在控制台打印图形 JSON','export file','保存 导出 文件'],
  ['import / import file / import json {JSON}','打开文件选择器，或从单行 JSON 导入画板','import file','导入 读取'],
  ['copy graph / copy ag2','复制当前图形数据或 AG2 输入参数','copy graph','复制 剪贴板'],
  ['doc / doc 名称','查看或修改当前文档名称','doc 我的几何图','重命名 文档'],
  ['console clear / console history','清空控制台输出或列出已输入命令','console history','日志 历史'],
  ['ui click 元素ID / ui set 元素ID 值','操作带 ID 的界面按钮、输入框或选项；作为界面动作通用入口','ui click fit-btn','键鼠 GUI'],
  ['自动命名规则','点 A-Z / P1 起补，直线 l1 起补，圆 c1 起补；自动输入名称若被占用会创建另一可用名称','point H 1 2','点名 补位 命名 递补'],
  ['多边形自动线段','polygon poly A B C 创建 sAB/sBC/sCA 等可供智能点/online 使用的边；点击首顶点闭合','polygon poly A B C','多边形 边 缩写 自动'],
  ['系统保留名','p_0、p_1、p_i、l_Re、l_Im、c_unit；这些是画布显示和控制台使用的同一对象名','select p_0','坐标轴 系统参考 单一命名'],
  ['shortcuts','查看 Ctrl+Z / Ctrl+Shift+Z / Delete / Esc / Space / Enter 等键盘功能和控制台等效命令','shortcuts','键盘 快捷键'],
 ]],
];
const normalize=s=>String(s||'').toLowerCase().normalize('NFKC').replace(/\s+/g,' ').trim();
const commands=sections.flatMap(([category,items])=>items.map(([syntax,description,example,keywords])=>({category,syntax,description,example,keywords})));
function registerMethods(methods){
 const groups={point:'点构造',line:'直线与射线',circle:'圆构造',polygon:'多边形'};
 for(const [group,title] of Object.entries(groups)){
  const entries=Object.entries(methods).filter(([,m])=>m.group===group).map(([id,m])=>{
   const inputs=m.inputs.map((x,i)=>`${i+1}.${{point:'点',line:'直线',circle:'圆',carrier:'直线或圆'}[x]||x}`).join('、');
   const description=`${m.label}。${inputs?'依次输入 '+inputs+'；':'无需已有对象；'}对象名称在首位。${id==='rotate'?'最后输入角度，可附加 rad；支持 2π/3。':id==='homothety'?'位似倍数可输入分数、平方根和 π。':id==='polygon'?'至少三个顶点；点击首顶点闭合，自动生成 sAB 等真实边线段。':''}`;
   return {category:'几何命令 · '+title,syntax:m.syntax,description,example:m.syntax,keywords:id+' '+m.label+' 作图 构造 内核'};
  });
  if(!entries.length)continue;
  sections.push(['几何命令 · '+title,entries.map(x=>[x.syntax,x.description,x.example,x.keywords])]);commands.push(...entries);
 }
}
function rank(entry,query){const q=normalize(query);if(!q)return 1;const fields=[entry.syntax,entry.description,entry.example,entry.keywords,entry.category].map(normalize);const words=q.split(/\s+/).filter(Boolean);let score=0;for(const w of words){let best=0;fields.forEach((f,i)=>{if(f===w)best=Math.max(best,100-i*10);else if(f.startsWith(w))best=Math.max(best,85-i*7);else if(f.includes(w))best=Math.max(best,65-i*6);else{const chars=[...w];let matched=0;for(const c of chars){if(f.includes(c))matched++;}if(w.length>=3&&matched>=w.length-1)best=Math.max(best,10+matched);}});if(!best)return 0;score+=best;}return score;}
function search(query){return commands.map(c=>({...c,score:rank(c,query)})).filter(c=>c.score>0).sort((a,b)=>b.score-a.score||commands.indexOf(commands.find(x=>x.syntax===a.syntax))-commands.indexOf(commands.find(x=>x.syntax===b.syntax)));}
function words(line){const tokens=String(line||'').match(/"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\S+/g)||[];return tokens.map(t=>(t[0]==='"'||t[0]==="'")&&t.length>1?t.slice(1,-1):t);}
return {sections,commands,search,words,registerMethods};
});
