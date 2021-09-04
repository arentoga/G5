/*QUERY PARA FILTRAR TUPLAS SEGUN CRITERIOS*/
select od.id_ofertadetalle,trim(od.descripcion_normalizada) as descripcion
from webscraping w inner join oferta o
on (w.id_webscraping=o.id_webscraping)
inner join oferta_detalle od
on (o.id_oferta=od.id_oferta)
left outer join ofertaperfil_tipo opt
on (od.ofertaperfil_id=opt.ofertaperfil_id)
where length(trim(od.descripcion_normalizada))<=320
and ( position('CURSO' in trim(descripcion_normalizada))>0
)