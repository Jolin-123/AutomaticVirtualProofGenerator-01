import React, { useEffect, useState} from 'react'
import axios from 'axios';
import './Student.css' ;
import { Link } from 'react-router-dom'



function ProductList() {

    const [data, setData] = useState([]);
  

//************** 1/27/2024 ************** ************** ************** **************
    // Working code
    // useEffect(() => {
    //     axios.get('http://localhost:8081/users')
    //     .then(res => {
    //       console.log("here is dataaaa:" + res.data); // Log the response data
    //       setData(res.data);     // Set the state with the response data
    //     })
    //     .catch(err => console.log(err));
    //   }, []);

    //************** 1/28/2024 ************** ************** ************** **************
    // Formating data to "YYYY-MM-DD" format
      function formatDate(dateString) {
        const date = new Date(dateString);
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    }

    useEffect(() => {
        axios.get('http://localhost:3030/productlist')
            .then(res => {
                const formattedData = res.data.map(item => ({
                    ...item,
                    start: formatDate(item.start), // Format start date
                    end: formatDate(item.end) // Format end date
                }));
                setData(formattedData);
            })
            .catch(err => console.log(err));
    }, []);

    
      const handleDelete = async (oid) => {
        try {
            await axios.delete(`http://localhost:3030/deleteProductList/${oid}`); //call the api
            window.location.reload();
        }catch(err){
            console.log("err from delete productList frontend @@ ");
        }
      }


    return (
        <div className='container_main'> 
            
             <div className='table_container'>
             <h2 className='order_list_title'> Order List </h2>

                <table className='list_table table_row'>  
                    <thead>
                    <tr>
                    <th>OrderID </th>
                    <th>BodyID </th>
                    <th>Customer </th>
                    <th>Label </th>
                    <th>GraphicID </th>

                    <th>Color </th>
                    <th>Team </th>
                    <th>Desc </th>
                    <th>XS </th>
                    <th>S </th>

                    <th>M </th>
                    <th>L </th>
                    <th>XL </th>
                    <th>XXL </th>
                    <th>XXXL </th>

                    <th>TTL </th>
                    <th>Start  </th>
                    <th>End  </th>
                    <th> <Link to="/create" className='btn btn-add'> Add+  </Link> </th>

                    </tr>
                    </thead>
                    
                    <tbody >
                    {
                        data.map((d, i) => (
                            <tr key={i} >
                                <td>{d.oid}</td>
                                <td>{d.customer}</td>
                                <td>{d.label}</td>
                                <td>{d.bid}</td>
                                <td>{d.gid}</td>
                                <td>{d.bcolor}</td>
                                <td>{d.team}</td>

                                <td>{d.bdesc}</td>
                                <td>{d.xs}</td>
                                <td>{d.s}</td>
                                <td>{d.m}</td>
                                <td>{d.l}</td>
                                <td>{d.xl}</td>

                                <td>{d.xxl}</td>
                                <td>{d.xxxl}</td>
                                <td>{d.ttl}</td>
                                <td>{d.start}</td>
                                <td>{d.end}</td>
                               
                             
                            <td className='button-container'>
                                <button className='btn read'> Read </button>
                                
                                <Link to={`/update/${d.oid}`}  className='btn update'> Update </Link>
                                <button className='btn delete' onClick={e=> handleDelete(d.oid)}> Delete </button>
                            </td>
                        </tr>
                      ))
                      }

                    </tbody>
                 </table>
             
             </div>
        </div>
    );
}

export default ProductList; 