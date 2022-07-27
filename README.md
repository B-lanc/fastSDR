# fastSDR
A faster way to calculate SDR than museval.metrics.bss_eval

museval.metrics.bss_eval takes a very long time to compute, this is especially prominent when only going for the SDR values, without the other metrics in mind. The fastsdr function inside fastsdr.py file can find the SDR values much faster with very low error.


### Results 
SDR is the np.nanmean of the calculated SDR using bss_eval, SDRtime is the time it takes to run the bss_eval function
fastsdr is the np.nanmean of the calculated SDR using fastsdr, fastsdr time is the tame it takes to run the fastsdr function

|SongName|SDR|SDRtime|fastsdr|fastsdrtime|
|--------|---|-------|-------|-----------|
|AM Contra - Heart Peripheral|5.134488617484768|151.27157252199686|5.134488424600735|0.3724633990059374|
|Al James - Schoolboy Facination|4.713653418273755|150.10335094299808|4.713653230096008|0.35549173499748576|
|Angels In Amplifiers - I'm Alright|5.024461470086696|89.51552404100221|5.024461340679623|0.3173046890005935|
|Arise - Run Run Run|5.9018282383673695|151.45320020799772|5.901828170539095|0.37044695900112856|
|BKS - Bulldozer|5.597891421591926|178.45870184899832|5.59789127917611|0.5871188769961009|
|BKS - Too Much|5.153163697513105|142.59593426700303|5.153163560573011|0.35735856999235693|
|Ben Carrigan - We'll Talk About It All Tonight|4.720654412003516|138.97276112700274|4.720654256880374|0.32208070599881466|
|Bobby Nobody - Stitch Up|4.647422788557435|146.6440289730017|4.647422586409636|0.37897697399603203|
|Buitraker - Revo X|5.378276259172617|151.7487010069999|5.378276079322002|0.4553843089961447|
|Carlos Gonzalez - A Place For Us|4.313983764842217|165.69053783799973|4.313983635730405|0.4807432819943642|
|Cristina Vane - So Easy|5.024856015221655|168.02146785600053|5.024855845974814|0.43500137400405947|
|Detsky Sad - Walkie Talkie|4.83677265532759|136.34712178900008|4.836772452936405|0.29280141099297907|
|Enda Reilly - Cur An Long Ag Seol|5.432690141333625|94.48663451999892|5.432690051472464|0.3334041479974985|
|Forkupines - Semantics|5.337890877232135|152.5019556139996|5.33789071513968|0.42274396699212957|
|Georgia Wonder - Siren|5.179500204694893|304.71538404199964|5.179500074605949|0.6685260090016527|
|Girls Under Glass - We Feel Alright|4.746446026631758|154.76429526300126|4.746445832052748|0.4830754110007547|
|Hollow Ground - Ill Fate|3.9812927416691286|79.01072658299745|3.9812926080145794|0.22346033398935106|
|James Elder & Mark M Thompson - The English Actor|5.054389437867685|136.9218701009995|5.054389289169408|0.3186013020022074|
|Juliet's Rescue - Heartbeats|6.3080423296316415|153.642137619001|6.308042163596951|0.4275981430109823|
|Little Chicago's Finest - My Own|7.311846179714789|155.0534170960018|7.311846066800439|0.4630734270031098|
|Louis Cressy Band - Good Time|5.923632875931011|143.07846589399924|5.923632736540521|0.4334246150101535|
|Lyndsey Ollard - Catching Up|5.224154307012139|146.83153739599948|5.224154188927059|0.3737288889969932|
|M.E.R.C. Music - Knockout|4.754747104065117|169.62662890900174|4.754746952078684|0.40084007898985874|
|Moosmusic - Big Dummy Shake|5.729606418969607|185.6378527449997|5.729606271806088|0.32504096800403204|
|Motor Tapes - Shore|5.132642543362842|211.39661621900086|5.132642375596803|0.38427183601015713|
|Mu - Too Bright|5.741069094883836|175.4570729870029|5.741068911946578|0.3869265289977193|
|Nerve 9 - Pray For The Rain|5.2967440529990295|178.22241842899894|5.296743944129308|0.5732581449992722|
|PR - Happy Daze|9.136936054582783|102.19452112199724|9.136935942666847|0.31427719500788953|
|PR - Oh No|5.609517633844266|49.88585270600015|5.609517555488855|0.1357586850062944|
|Punkdisco - Oral Hygiene|8.335911207472625|161.00886545100002|8.335911059338185|0.34041560599871445|
|Raft Monk - Tiring|3.75427680579437|160.21856046600078|3.7542766845265847|0.35502234898740426|
|Sambasevam Shanmugam - Kaathaadi|5.949107349760848|142.83706377600174|5.9491072324419205|0.411010776006151|
|Secretariat - Borderline|3.465210345472199|153.3948036740003|3.4652102093137467|0.4649616369861178|
|Secretariat - Over The Top|4.10353885789089|144.06330657399667|4.103538698497384|0.38009198500367347|
|Side Effects Project - Sing With Me|7.175289073746167|158.28074183900026|7.17528892133689|0.42069215499213897|
|Signe Jakobsen - What Have You Done To Me|5.052667760876627|95.80871894499796|5.052667616035138|0.30333559500286356|
|Skelpolu - Resurrection|4.082065158629926|268.83016313199914|4.082065067942375|0.5222668979986338|
|Speak Softly - Broken Man|5.552485163107511|147.99263738599984|5.552485033526625|0.3970112539973343|
|Speak Softly - Like Horses|5.723475803694948|144.93154214300012|5.723475631696722|0.4901106349861948|
|The Doppler Shift - Atrophy|5.373423543401158|181.77653529899908|5.373423432267869|0.5737287579977419|
|The Easton Ellises (Baumi) - SDRNR|7.743881463723608|156.67262716099867|7.743881377858391|0.4208930510067148|
|The Easton Ellises - Falcon 69|5.172894775330021|155.74558180099848|5.1728946521163|0.4157258630002616|
|The Long Wait - Dark Horses|4.2255161335786715|158.14514869599952|4.22551600920717|0.4905217129999073|
|The Mountaineering Club - Mallory|4.23781622717368|140.62956723599927|4.237816051426757|0.34141442899999674|
|The Sunshine Garcia Band - For I Am The Moon|5.461904126022936|172.02792842200142|5.4619039373725435|0.5104596820019651|
|Timboz - Pony|3.8545467827919806|166.4732786860004|3.8545466193163795|0.4355094009952154|
|Tom McKenzie - Directions|5.0772375781376695|78.82977491299971|5.077237380796387|0.26384307999978773|
|Triviul feat. The Fiend - Widow|6.1940731879771205|153.6455953270015|6.194073054752564|0.3902430029993411|
|We Fell From The Sky - Not You|6.064584791422877|141.1617287529989|6.0645846016550955|0.329197409999324|
|Zeno - Signs|4.291883081255952|154.8784268390009|4.291882985480038|0.4001308059960138|
