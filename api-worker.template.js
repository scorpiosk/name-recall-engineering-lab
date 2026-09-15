/* Browser and API execute the same checkpoint and arithmetic. */
/* ENGINE_BUNDLE */
/* PAYLOAD_BUNDLE */
/* PAGE_BUNDLE */
const json=(body,status=200)=>new Response(JSON.stringify(body),{status,headers:{'content-type':'application/json; charset=utf-8','cache-control':'no-store','x-content-type-options':'nosniff'}});
export default {
 async fetch(request){
  const url=new URL(request.url);
  if(url.pathname==='/api/health')return json({status:'ok',model:'name-recall-5l',parameters:MODEL.metadata.parameters,runtime:'JavaScript CPU',persistence:'none'});
  if(url.pathname==='/api/predict'){
   if(request.method!=='POST')return json({error:'Use POST with a JSON request body.'},405);
   const origin=request.headers.get('origin');if(origin&&origin!==url.origin)return json({error:'Cross-origin requests are not accepted.'},403);
   if(!request.headers.get('content-type')?.includes('application/json'))return json({error:'Content-Type must be application/json.'},415);
   if(Number(request.headers.get('content-length')||0)>12000)return json({error:'Request too large.'},413);
   try{const raw=await request.text();if(raw.length>12000)return json({error:'Request too large.'},413);const body=JSON.parse(raw);const result=EngineeringCore.generate(MODEL,body);delete result.trace;return json(result)}catch(e){return json({error:e.message||'Invalid request.'},400)}
  }
  if((url.pathname==='/'||url.pathname==='/index.html')&&(request.method==='GET'||request.method==='HEAD'))return new Response(request.method==='HEAD'?null:PAGE,{headers:{'content-type':'text/html; charset=utf-8','cache-control':'no-cache','x-content-type-options':'nosniff'}});
  return new Response('Not found',{status:404});
 }
};
