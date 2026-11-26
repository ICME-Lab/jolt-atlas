#include <metal_stdlib>
using namespace metal;
using U = uint4;
constant U prime = U(0xffffffbb, 0xffffffff, 0xffffffff, 0x007fffff);
inline U subraw(U a, U b) {
    ulong c = 0;
    U out;
    for (uint i=0;i<4;i++) {
        ulong z = (ulong)a[i]-(ulong)b[i]-c;
        out[i] = (uint)z;
        c = z >> 63;
    }
    return out;
}
inline U addraw(U a, U b) {
    ulong c = 0;
    U out;
    for (uint i=0;i<4;i++) {
        c += (ulong)a[i]+b[i];
        out[i] = (uint)c;
        c >>= 32;
    }
    return out;
}
inline bool ge(U a, U b) {
    if (a.w != b.w) return a.w>b.w;
    if (a.z != b.z) return a.z>b.z;
    if (a.y != b.y) return a.y>b.y;
    return a.x>=b.x;
}
inline U norm(U a) { return ge(a,prime) ? subraw(a,prime) : a; }
inline U add(U a,U b) { return norm(addraw(a,b)); }
inline U sub(U a,U b) { return ge(a,b) ? subraw(a,b) : subraw(addraw(a,prime),b); }
inline U mul(U a,U b) {
    uint v[8] = {};
    #pragma unroll
    for (uint i=0;i<4;i++) {
        ulong c = 0;
        #pragma unroll
        for (uint j=0;j<4;j++) {
            c += (ulong)a[i]*b[j] + v[i+j];
            v[i+j] = (uint)c;
            c >>= 32;
        }
        v[i+4] = (uint)c;
    }
    U lo = U(v[0],v[1],v[2],v[3]&0x007fffff);
    U hi;
    #pragma unroll
    for (uint i=0;i<4;i++) hi[i]=(v[i+3]>>23)|(v[i+4]<<9);
    ulong carry = 0;
    #pragma unroll
    for (uint i=0;i<4;i++) {
        carry += (ulong)hi[i]*69 + lo[i];
        lo[i] = (uint)carry;
        carry >>= 32;
    }
    uint top = lo.w >> 23;
    lo.w &= 0x007fffff;
    return norm(addraw(lo,U(top*69,0,0,0)));
}
constant uint operation [[function_constant(0)]];

struct Args {
    U r, coeff[7];
    uint offset[12];
    uint old_len, len, width, stride, rows, cols, kind, flags;
    uint bits, size, stage, stored, axis_stride;
};

struct Input {
    device const uint *words;
    device const uint4 *tiles;
    device const U *axis, *table;
    constant Args &a;
};

inline U signed_word(uint low, uint high) {
    if ((high >> 31) == 0) return U(low, high, 0, 0);
    return addraw(prime, U(low, high, 0xffffffff, 0xffffffff));
}
inline U signed_small(int value) { return signed_word(uint(value), uint(value >> 31)); }

inline uint code(device const uint *words, uint offset, uint i, uint bits) {
    uint bit = i * bits;
    uint ix = offset + (bit >> 5);
    ulong pair = ulong(words[ix]) | (ulong(words[ix+1]) << 32);
    return uint(pair >> (bit & 31)) & uint((1ul << bits) - 1);
}

inline void witness(thread const Input &s, uint i, thread U *v) {
    constant Args &a=s.a;
    if(a.stage >= 2) {
        auto dense=reinterpret_cast<device const U *>(s.words);
        for(uint j=0;j<a.stored;j++) v[j]=dense[j*a.old_len+i];
        return;
    }
    uint row=i/a.stride, col=i%a.stride;
    if(operation==0) {
        uint index=0;
        bool valid=col<=row;
        v[0]=U(0); v[4]=U(0);
        if(valid) {
            uint diff=(row^col)>>6, level=diff ? 31-clz(diff) : 0, diagonal=a.rows>>6;
            uint id=diff ? 2*diagonal-(diagonal>>level)+(row>>(level+7)) : row>>6;
            uint4 t=s.tiles[id];
            uint local=(row&(t.x-1))*t.x+(col&(t.x-1));
            index=code(s.words,t.z,local,a.bits);
            v[0]=signed_small(as_type<int>(s.words[t.y+local]));
            if(t.w!=0xffffffff) v[4]=signed_word(s.words[t.w+2*local],s.words[t.w+2*local+1]);
        }
        v[1]=U(index,0,0,0);
        v[2]=s.table[2*index];
        v[3]=s.table[2*index+1];
        v[5]=U(valid,0,0,0);
    } else if(operation==1) {
        bool valid=row<a.rows && col<a.cols;
        uint ix=row*a.cols+col;
        int x=valid?as_type<int>(s.words[ix]):0;
        uint index=code(s.words,a.rows*a.cols*(1+(a.flags&1)),i,a.bits);
        long magnitude=abs(long(x));
        long last=long(a.size-1);
        long shifted=long(x)-long(as_type<int>(a.coeff[5].x));
        bool symmetric=a.flags&2;
        v[0]=signed_small(x);
        v[1]=U(index,0,0,0);
        v[2]=s.table[index];
        v[3]=U(valid&&x>=0,0,0,0);
        v[4]=U(valid&&!symmetric?uint(magnitude):0,0,0,0);
        v[5]=U(valid&&!symmetric?uint(max(-shifted,0l)):0,0,0,0);
        v[6]=U(valid?uint(symmetric?max(magnitude-last,0l):max(shifted-last,0l)):0,0,0,0);
        if(a.flags&1) v[7]=signed_small(valid?as_type<int>(s.words[a.rows*a.cols+ix]):0);
    } else {
        uint key=code(s.words,0,i,a.bits);
        for(uint j=0;j<a.stored;j++) {
            uint shift=s.tiles[j].x, mask=s.tiles[j].y, start=s.tiles[j].z;
            v[j]=s.table[start+((key>>shift)&mask)];
        }
    }
}

inline void public_columns(thread const Input &s, uint i, thread U *v) {
    constant Args &a=s.a;
    uint row=i/a.width, col=i%a.width;
    auto axis=s.axis;
    if(operation==0) {
        v[6]=axis[a.offset[0]+row];
        v[7]=axis[a.offset[1]+row];
        v[8]=axis[a.offset[2]+row];
        v[9]=axis[a.offset[3]+col];
        v[10]=mul(axis[a.offset[4]+row],axis[a.offset[5]+col]);
        v[11]=mul(axis[a.offset[6]+row],axis[a.offset[7]+col]);
        v[12]=axis[a.offset[8]+row];
    } else if(operation==1) {
        v[a.stored]=mul(axis[a.offset[0]+row],axis[a.offset[1]+col]);
        v[a.stored+1]=mul(axis[a.offset[2]+row],axis[a.offset[3]+col]);
        v[a.stored+2]=mul(axis[a.offset[4]+row],axis[a.offset[5]+col]);
    } else {
        v[a.stored]=mul(axis[a.offset[0]+row],axis[a.offset[1]+col]);
    }
}

inline U formula(thread const U *v, thread const Input &s) {
    constant Args &a=s.a;
    U one=U(1,0,0,0);
    if(operation==0) {
        U mix=a.coeff[0], last=a.coeff[1], radix=a.coeff[2], limit=a.coeff[3];
        U clamp=add(sub(sub(sub(v[6],v[0]),v[1]),v[4]),mul(mul(mix,v[4]),sub(v[1],last)));
        U totals=sub(mul(v[5],v[3]),mul(v[9],sub(sub(limit,v[7]),mul(radix,add(add(limit,one),v[8])))));
        return add(mul(v[5],add(mul(v[10],v[2]),mul(v[11],clamp))),mul(v[12],totals));
    }
    if(operation==1) {
        bool symmetric=a.flags&2;
        U magnitude=symmetric?add(v[1],v[6]):v[4];
        U signed_eq=sub(v[0],mul(sub(add(v[3],v[3]),one),magnitude));
        U last_minus_index=sub(a.coeff[4],v[1]);
        U clamp;
        if (symmetric) clamp=mul(a.coeff[1],mul(v[6],last_minus_index));
        else {
            U x=sub(add(sub(sub(v[0],a.coeff[6]),v[1]),v[5]),v[6]);
            clamp=add(mul(a.coeff[1],x),add(mul(a.coeff[2],mul(v[5],v[1])),mul(a.coeff[3],mul(v[6],last_minus_index))));
        }
        U out=add(mul(v[3],magnitude),v[2]);
        if(a.flags&1) out=mul(out,v[7]);
        return mul(v[a.stored],add(mul(v[a.stored+1],out),mul(v[a.stored+2],add(mul(a.coeff[0],signed_eq),clamp))));
    }
    U product=v[a.stored];
    for(uint j=0;j<a.stored;j++) product=mul(product,v[j]);
    return mul(s.axis[a.axis_stride-1],product);
}

inline uint degree(constant Args &a) {
    return operation==0 ? 4 : (operation==1 ? 4+(a.flags&1) : a.stored+1);
}

kernel void round_sum(device const uint *input [[buffer(0)]], device U *write [[buffer(1)]],
                      device U *result [[buffer(2)]], constant Args &a [[buffer(3)]],
                      device const uint4 *tiles [[buffer(4)]], device const U *axis [[buffer(5)]],
                      device const U *table [[buffer(6)]],
                      uint2 position [[thread_position_in_grid]], uint2 threadPos [[thread_position_in_threadgroup]],
                      uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                      uint simd [[simdgroup_index_in_threadgroup]], uint2 groupSize [[threads_per_threadgroup]],
                      uint2 groupGrid [[threadgroups_per_grid]]) {
    uint i=position.x, local=threadPos.x;
    threadgroup U shared[128];
    U sums[10]={};
    uint mid=a.len/2, count=degree(a), columns=operation==0?13:a.stored+(operation==1?3:1);
    if(i<mid){
        uint input_stride = a.stage < 2 ? a.cols : 4*a.stored*a.old_len;
        Input source{input+group.y*input_stride,tiles,axis+group.y*a.axis_stride,table,a};
        write+=group.y*a.stored*a.len;
        U lo[13],hi[13],delta[13];
        witness(source,i,lo);
        witness(source,i+mid,hi);
        if(a.stage!=0) {
            U x[13],y[13];
            witness(source,i+a.len,x);
            witness(source,i+mid+a.len,y);
            for(uint j=0;j<a.stored;j++) {
                lo[j]=add(lo[j],mul(a.r,sub(x[j],lo[j])));
                hi[j]=add(hi[j],mul(a.r,sub(y[j],hi[j])));
                write[j*a.len+i]=lo[j];
                write[j*a.len+i+mid]=hi[j];
            }
        }
        public_columns(source,i,lo);
        public_columns(source,i+mid,hi);
        for(uint j=0;j<columns;j++) delta[j]=sub(hi[j],lo[j]);
        sums[0]=formula(lo,source);
        for(uint t=1;t<=count;t++){
            for(uint j=0;j<columns;j++) lo[j]=add(lo[j],delta[j]);
            if(t>=2) sums[t-1]=formula(lo,source);
        }
    }
    for(uint t=0;t<count;t++){
        U v=sums[t];
        for(uint step=16;step;step/=2) v=add(v,simd_shuffle_down(v,(ushort)step));
        if(lane==0) shared[t*8+simd]=v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(local<count){
        U v=U(0);
        for(uint j=0;j<groupSize.x/32;j++) v=add(v,shared[local*8+j]);
        result[(group.y*groupGrid.x+group.x)*count+local]=v;
    }
}

inline ulong2 clmul(ulong a, ulong b) {
    ulong2 bases[4] = {ulong2(a,0),ulong2(a<<1,a>>63),ulong2(a<<2,a>>62),ulong2(a<<3,a>>61)};
    ulong2 carryless=ulong2(0);
    for (int i=60;i>=0;i-=4) {
        carryless=ulong2(carryless.x<<4,(carryless.y<<4)|(carryless.x>>60));
        ulong nibble=b>>i;
        for (uint j=0;j<4;j++) carryless^=bases[j]&ulong2(0ul-((nibble>>j)&1));
    }
    return carryless;
}

inline ulong2 gf128(ulong2 a, ulong2 b) {
    auto lo=clmul(a.x,b.x);
    auto hi=clmul(a.y,b.y);
    auto middle=clmul(a.x^a.y,b.x^b.y)^lo^hi;
    ulong x=hi.x^middle.y,y=hi.y;
    ulong spill=(y>>63)^(y>>62)^(y>>57);
    return ulong2(
        lo.x^x^(x<<1)^(x<<2)^(x<<7)^spill^(spill<<1)^(spill<<2)^(spill<<7),
        lo.y^middle.x^(x>>63)^(x>>62)^(x>>57)^y^(y<<1)^(y<<2)^(y<<7)
    );
}

kernel void bitz_tree(device const ulong2 *input [[buffer(0)]], device ulong2 *output [[buffer(1)]],
                      constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i<count) output[i]=gf128(input[i],input[i+count]);
}

struct BitzArgs {
    ulong2 challenge;
    uint len, low_size, binding, at_one;
};

kernel void bitz_round(device const ulong2 *input [[buffer(0)]], device ulong2 *output [[buffer(1)]],
                       device ulong2 *partial [[buffer(2)]], constant BitzArgs &a [[buffer(3)]],
                       device const ulong2 *low [[buffer(4)]], device const ulong2 *high [[buffer(5)]],
                       uint i [[thread_position_in_grid]], uint local [[thread_position_in_threadgroup]],
                       uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                       uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong2 sums[8];
    ulong2 u=ulong2(0), v=ulong2(0);
    uint mid=a.len/2;
    if (i<mid) {
        uint right=a.len*(1+a.binding);
        auto l0=input[i],l1=input[i+mid],r0=input[right+i],r1=input[right+i+mid];
        if (a.binding) {
            l0^=gf128(a.challenge,input[i+a.len]^l0);
            l1^=gf128(a.challenge,input[i+mid+a.len]^l1);
            r0^=gf128(a.challenge,input[right+i+a.len]^r0);
            r1^=gf128(a.challenge,input[right+i+mid+a.len]^r1);
            output[i]=l0; output[i+mid]=l1; output[a.len+i]=r0; output[a.len+i+mid]=r1;
        }
        auto weight=low[i&(a.low_size-1)];
        if(mid>a.low_size) weight=gf128(weight,high[i/a.low_size]);
        u=gf128(weight,gf128(a.at_one?l1:l0,a.at_one?r1:r0));
        v=gf128(weight,gf128(l0^l1,r0^r1));
    }
    for(uint step=16;step;step/=2) {
        // The shuffle itself is 32-bit on Apple GPUs.
        auto uu=simd_shuffle_down(as_type<uint4>(u),(ushort)step);
        auto vv=simd_shuffle_down(as_type<uint4>(v),(ushort)step);
        u^=as_type<ulong2>(uu); v^=as_type<ulong2>(vv);
    }
    if(lane==0) {sums[simd]=u;sums[4+simd]=v;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(local==0) {
        partial[2*group]=sums[0]^sums[1]^sums[2]^sums[3];
        partial[2*group+1]=sums[4]^sums[5]^sums[6]^sums[7];
    }
}
