// C++ of Paul McNamee's cross-doc Coref Java code.

// Miles Osborne

#include<string>
#include<iostream>
#include<boost/algorithm/string.hpp>
#include<gflags/gflags.h>

using namespace std;

DEFINE_int32(tables, 70, "Number of LSH tables"); //should be
DEFINE_string(stopWords, "/home/miles/projects/fsd/data/stop.txt", "Stop word list");
DEFINE_bool(oneHit, false, "Assume each update only matches with document");

class Interner {
public:
  Interner(){
    hMap_ = new map<std::string, int>;
    revMap_ = new map<int, std::string>;
    startId_ = 1;
  }
  int get(std::string key){
    std::map<std::string, int>::iterator it = hMap_->find(key);
    if (it != hMap_->end()){
      return it->second;
    } else {
      int next = startid;
      hMap_[key] = startid;
      revMap_[next] = key;
      startid_++;
      return next;
    }
  }
  std::string revGet(int key){
    std::map<int, std::string>::iterator it = revMap_->find(key);
    if (it!= revMap_->end()){
      return it->second;
    } else {
      return "";
    }
  }  
  std::map<std::string, int> hMap_*;
  std::map<int, std::string> revMap_*;
private:
  int startId = 1;
};

class FeatMap {
public:
  std::map<int, int> hMap_;
  intern_ = new Interner();
  int number1 = 1;
  FeatMap(){
    hMap_ = new map<int, int>;
  }
  FeatMap(Interner intern){
    hMap_ = new map<int, int>;
    intern_ = intern;
  }
  int get(std::string key){
    return hMap_[inter[key]];
  }
  int put(std::string k, int i){
    int prev = intern.get(k);
    hMap_[intern.get(k),i];
    return prev;//prob wrong
  }
  void addOne(std::string k){
    int ki = intern.get(k);
    int i = hMap.get(ki);
    if (i == NULL){//looks wrong
      hMap.put(ki, 1);
    } else {
      hMap.put(ki, i + 1);
    }
  }
  void addAll(featMap fm){
    map<int, int>::iterator end = fm.hMap.end();
    for(map<int, int>::iterator it = fm.hMap.begin(); it!= end; ++it){
      addValue(hMap, it->first, it->second);
    }
  }
  void addValue(map<int, int> hMap, int key, int value){
    int i = hMap.get(key);
    if (i == NULL){//fix
      hMap.put(key, value);
    } else {
      hMap.put(key, i + value);
    }
  }
  bool startsWithPrefix(std::string str, std::string prefix){
    bool found=false;
    int size = prefix.size();
    if (size > str.size()){
      return false;
    }
    for(int i=0;i<size;i++){
      if (str[i] != prefix[i]){
	return false;
      }
    }
    return true;
  }
  vector<std::string> featureWithPrefix(std::string prefix, bool actuallyWithout){
    vector<string> list;
    std::map<int, int>::iterator end = hMap.end();
    for(map<int, int>::iterator it = hMap.begin(); it!= end;++it){
      std::string key = intern.revGet(it->first);
      if (actuallyWithout && ! startsWithPrefix(key, prefix)){
	key = key.substr(key.find(":",0)+1, key.size());
	list.push_back(key);
      }
      if (!actuallyWithout && startsWith(key, prefix)){
	key = key.substr(key.find(":",0)+1, key.size());
	list.push_back(key);
      }
    }
    return list;
  }
  void print(){
    std::map<int, int>::iterator end = hMap.end();
    for(std::map<int, int>::iterator it = hMap.start(); it != end; ++it){
      std::cout << intern.revGet(it->first) << "\t" << it->second << "\n";
    }
  }
  std::string toString(){
    std::string result;
    std::map<int, int>::iterator end = hMap.end();
    for(std::map<int, int>::iterator it = hMap.start(); it != end; ++it){
      result = result + intern.revGet(it->first) + it->second + "\n";
    }
    return result;
  }
};
class DocEntity {
  std::string eid_;
  std::string docid_;
  std::string type_;
  FeatMap features_;
  std::string lc(std::str){
    std::res = str;
    int size = str.size();
    for(int i=0;i<size;i++){
      res[i] = tolower(res[i]);
    }
    return res;
  }  
  DocEntity(std::string eid, std::string docid, string type){
    eid_ = eid;
    docid_ = docid;
    type_ = lc(type);
    features_ = new FeatMap();
  }
  std::string normalise(std::string s){
    return lc(s);
  }
  void addFeature(std::string prefix, std::string featName){
    features_.addOne(prefix + normalise(featName));
  }
  int lastIndex(char c, std:;string eid){
    int size = eid.size();
    for(int i=size-1, i>0, i--){
      if (eid[i] == c){
	return i;
      }
    }
    return -1;//error
  }
  std::string docIdfromEid(std::string eid){
    int idx = lastIndex('_', eid); assert(idx != -1);
    std::string prefix = eid.substr(0, idx);
  }
  map<string, DocEntity> loadKB(std::string fName){
    std::map<std::string, DocEntity> dles;
    std::map<string, std::map<std::string, bool> entsInDoc;
    
class Cluster {
  std::vector<DocEntity> entities_;
  std::set<std::string> normNameSet_;//hash set of normalised names
  std::vector<std::string> rawnames_; // hash set of raw names
  FeatMap feats;
  std::map<string, bool> dids; // hash set of docids
  Cluster(DocEntity docent){
    entities_ = new std::vector<DocEntity>;
    entities_.psh_back(docent);
  }
};

class EntityStore {
public:
  std::map<int, Cluster> hmap_;
};

main(int argc, char *argv[]){
  //parse command line options
  google::ParseCommandLineFlags(&argc, &argv, true);
  while(!cin.eof()){
    ;
  }
} 
