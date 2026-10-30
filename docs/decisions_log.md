//config\_rev2.yaml

//selection setting

This is used for result\_rev2



selection:

&#x20; min\_phishing\_recall: 0.89

&#x20; max\_legit\_fpr: 0.03



//dedup setting



dedup:

&#x20; near\_dup\_threshold: 0.90

&#x20; review\_sample\_size: 200



//config\_rev3.yaml



Revision 3 (Masking): Applied a 92-term mask list to text.mask\_terms. This removes severe source leakage identified in the Revision 2 audit, including Enron employee names/jargon, parsing bugs (e.g., "body url"), and Nazario dataset artifacts (e.g., hex codes like "0x0"). Thresholds kept at 0.89 phishing recall and 0.03 legit FPR.



//config\_rev4.yaml



&#x20;strip\_urls: false       

&#x20; strip\_email\_addresses: false

without masked terms



//config\_rev5.yaml



masked terms is available such as revision 3

strip\_urls: false

&#x20; strip\_email\_addresses: false



