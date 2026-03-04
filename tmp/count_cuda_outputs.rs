use vector_ta::indicators::registry::list_indicators;
fn main(){
 let mut total=0;let mut one=0;let mut multi=0;
 for i in list_indicators(){
   if i.capabilities.supports_cuda_batch{ total+=1; if i.outputs.len()==1 {one+=1}else{multi+=1;println!("{}:{}", i.id, i.outputs.len());}}
 }
 println!("total={} one={} multi={}", total, one, multi);
}
